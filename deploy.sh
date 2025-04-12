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
ExecStart=/home/ec2-user/mosaic_generator/venv/bin/gunicorn --workers 3 --bind unix:mosaic-generator.sock -m 007 wsgi:app

[Install]
WantedBy=multi-user.target
EOF

# Remove any existing Nginx configurations
sudo rm -f /etc/nginx/conf.d/*.conf

# Configure Nginx
sudo tee /etc/nginx/conf.d/mosaic-generator.conf << EOF
server {
    listen 80 default_server;
    server_name localhost;

    location / {
        proxy_pass http://unix:/home/ec2-user/mosaic_generator/mosaic-generator.sock;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }

    location /static {
        alias /home/ec2-user/mosaic_generator/static;
    }
}
EOF

# Create necessary directories
mkdir -p static/uploads
chmod 755 static/uploads

# Set proper permissions
sudo chown -R ec2-user:nginx /home/ec2-user/mosaic_generator
sudo chmod -R 755 /home/ec2-user/mosaic_generator

# Start and enable services
sudo systemctl start mosaic-generator
sudo systemctl enable mosaic-generator
sudo systemctl restart nginx
sudo systemctl enable nginx 