#!/bin/bash

# Update system packages
sudo yum update -y

# Install required system packages
sudo yum install -y python3 python3-pip python3-devel gcc nginx

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
pip install gunicorn

# Create systemd service file
sudo tee /etc/systemd/system/mosaic-generator.service << EOF
[Unit]
Description=Gunicorn instance to serve mosaic generator
After=network.target

[Service]
User=ec2-user
Group=nginx
WorkingDirectory=/home/ec2-user/mosaic_generator
Environment="PATH=/home/ec2-user/mosaic_generator/venv/bin"
Environment="PYTHONUNBUFFERED=1"
Environment="FLASK_SECRET_KEY=$(openssl rand -hex 32)"
ExecStart=/home/ec2-user/mosaic_generator/venv/bin/gunicorn --workers 3 --bind unix:/run/gunicorn/mosaic-generator.sock -m 007 wsgi:app --log-level debug --timeout 0
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Remove any existing Nginx configurations
sudo rm -f /etc/nginx/conf.d/*.conf
sudo rm -f /etc/nginx/nginx.conf

# Create main Nginx configuration
sudo tee /etc/nginx/nginx.conf << EOF
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log debug;
pid /run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    log_format  main  '\$remote_addr - \$remote_user [\$time_local] "\$request" '
                      '\$status \$body_bytes_sent "\$http_referer" '
                      '"\$http_user_agent" "\$http_x_forwarded_for"';

    access_log  /var/log/nginx/access.log  main;

    sendfile            on;
    tcp_nopush          on;
    tcp_nodelay         on;
    keepalive_timeout   65;
    types_hash_max_size 4096;

    # Increase buffer sizes for large file uploads
    client_max_body_size 300M;
    client_body_buffer_size 128k;
    client_header_buffer_size 1k;
    large_client_header_buffers 4 4k;

    include             /etc/nginx/mime.types;
    default_type        application/octet-stream;

    include /etc/nginx/conf.d/*.conf;
}
EOF

# Configure application Nginx
sudo tee /etc/nginx/conf.d/mosaic-generator.conf << EOF
server {
    listen 80;
    server_name _;
    root /home/ec2-user/mosaic_generator;

    # Increase timeout for large file uploads
    client_max_body_size 300M;
    client_body_timeout 3600s;
    client_header_timeout 3600s;
    keepalive_timeout 3600s;
    send_timeout 3600s;

    location / {
        proxy_pass http://unix:/run/gunicorn/mosaic-generator.sock;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Increase proxy timeouts
        proxy_connect_timeout 3600s;
        proxy_send_timeout 3600s;
        proxy_read_timeout 3600s;
    }

    location /static {
        alias /home/ec2-user/mosaic_generator/static;
    }

    # Error pages
    error_page 500 502 503 504 /50x.html;
    location = /50x.html {
        root /usr/share/nginx/html;
    }
}
EOF

# Create necessary directories with proper permissions
sudo mkdir -p /home/ec2-user/mosaic_generator/static/uploads
sudo mkdir -p /home/ec2-user/mosaic_generator/static/templates
sudo chown -R ec2-user:nginx /home/ec2-user/mosaic_generator
sudo chmod -R 775 /home/ec2-user/mosaic_generator

# Ensure the socket directory exists and has correct permissions
sudo mkdir -p /run/gunicorn
sudo chown ec2-user:nginx /run/gunicorn
sudo chmod 775 /run/gunicorn

# Remove any existing socket file
sudo rm -f /run/gunicorn/mosaic-generator.sock
sudo rm -f /home/ec2-user/mosaic_generator/mosaic-generator.sock

# Add nginx user to ec2-user group
sudo usermod -a -G nginx ec2-user
sudo usermod -a -G ec2-user nginx

# Reload systemd to pick up new service file
sudo systemctl daemon-reload

# Start and enable services
sudo systemctl restart mosaic-generator
sudo systemctl enable mosaic-generator
sudo systemctl restart nginx
sudo systemctl enable nginx

# Check service status
echo "Checking service status..."
sudo systemctl status mosaic-generator
sudo systemctl status nginx

# Check socket permissions
echo "Checking socket permissions..."
ls -l /run/gunicorn/mosaic-generator.sock

# Check directory permissions
echo "Checking directory permissions..."
ls -la /home/ec2-user/mosaic_generator/static

# Test Nginx configuration
echo "Testing Nginx configuration..."
sudo nginx -t 