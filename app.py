import os
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter
import pandas as pd
import qrcode
import base64
import psutil
import math
import tempfile
import shutil
import uuid
import schedule
import time
import threading
from datetime import datetime, timedelta

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'static/uploads')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def cleanup_upload_folder():
    """Clean up files older than 24 hours from the upload folder"""
    try:
        current_time = datetime.now()
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                # Get file's last modification time
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                # If file is older than 24 hours, delete it
                if current_time - file_time > timedelta(hours=24):
                    os.remove(file_path)
                    print(f"Deleted old file: {filename}")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def run_scheduler():
    """Run the scheduler in a separate thread"""
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

# Schedule the cleanup job to run daily at midnight
schedule.every().day.at("00:00").do(cleanup_upload_folder)

# Start the scheduler in a background thread
scheduler_thread = threading.Thread(target=run_scheduler)
scheduler_thread.daemon = True
scheduler_thread.start()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'xlsx', 'xls'}

def get_unique_filename(filename):
    """Generate a unique filename using UUID"""
    ext = filename.rsplit('.', 1)[1].lower()
    return f"{uuid.uuid4()}.{ext}"

def cleanup_old_files():
    """Remove old files from the upload folder"""
    try:
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"Error cleaning up files: {e}")

def estimate_image_memory(image):
    """Estimate memory usage of an image in bytes"""
    # Each pixel uses 4 bytes (RGBA) or 3 bytes (RGB)
    bytes_per_pixel = 4 if image.mode == 'RGBA' else 3
    return image.width * image.height * bytes_per_pixel

def check_memory_usage(image, max_memory_mb=500):
    """Check if image memory usage exceeds limit"""
    max_memory_bytes = max_memory_mb * 1024 * 1024
    current_memory = estimate_image_memory(image)
    return {
        'exceeds_limit': current_memory > max_memory_bytes,
        'current_memory_mb': current_memory / (1024 * 1024),
        'max_memory_mb': max_memory_mb
    }

def estimate_final_image_memory(width_inches, height_inches, dpi, margin=0):
    """Estimate memory usage of the final image in bytes"""
    # Convert inches to pixels
    width_pixels = int(width_inches * dpi)
    height_pixels = int(height_inches * dpi)
    
    # Add margin
    width_pixels += margin * 2
    height_pixels += margin * 2
    
    # Each pixel uses 3 bytes (RGB)
    return width_pixels * height_pixels * 3

def validate_memory_requirements(width_inches, height_inches, dpi, margin=0, max_memory_mb=1500):
    """Validate if the image can be generated within memory constraints"""
    estimated_memory = estimate_final_image_memory(width_inches, height_inches, dpi, margin)
    max_memory_bytes = max_memory_mb * 1024 * 1024
    
    return {
        'can_process': estimated_memory <= max_memory_bytes,
        'estimated_memory_mb': estimated_memory / (1024 * 1024),
        'max_memory_mb': max_memory_mb
    }

def generate_qr_mosaic(image_path, excel_path, num_cols, num_rows, square_tiles, tile_size, 
                      margin=0, qr_padding=0, qr_opacity=100, width_inches=0, height_inches=0, dpi=300, bg_opacity=100, preview=False):
    # Load Excel data
    df = pd.read_excel(excel_path)
    
    # Validate memory requirements for full-size generation
    if not preview:
        memory_validation = validate_memory_requirements(width_inches, height_inches, dpi, margin)
        if not memory_validation['can_process']:
            raise ValueError(
                f"Image size ({memory_validation['estimated_memory_mb']:.2f}MB) exceeds available memory "
                f"({memory_validation['max_memory_mb']}MB). Please reduce DPI or physical dimensions."
            )
    
    # Load the base image
    base_image = Image.open(image_path)
    base_width, base_height = base_image.size

    # If in preview mode, resize the base image to a smaller size
    if preview:
        preview_size = 500  # Maximum dimension for preview
        if base_width > base_height:
            new_width = preview_size
            new_height = int(base_height * (preview_size / base_width))
        else:
            new_height = preview_size
            new_width = int(base_width * (preview_size / base_height))
        base_image = base_image.resize((new_width, new_height), resample=Image.LANCZOS)
        base_width, base_height = base_image.size
        # Adjust DPI for preview
        dpi = 72  # Lower DPI for preview
    else:
        # Check memory usage but don't resize
        memory_info = check_memory_usage(base_image)
        if memory_info['exceeds_limit']:
            print(f"Warning: Image memory usage ({memory_info['current_memory_mb']:.2f}MB) exceeds limit ({memory_info['max_memory_mb']}MB)")

    # If inches are specified, convert to pixels using DPI
    if width_inches > 0 and height_inches > 0:
        # Convert inches to pixels using DPI
        base_width = int(width_inches * dpi)
        base_height = int(height_inches * dpi)
        base_image = base_image.resize((base_width, base_height), resample=Image.LANCZOS)
    
    # Determine the tile size
    if tile_size:
        # Use the tile size provided by the user
        tile_size = tile_size
    elif square_tiles:
        # Force square tiles using the smaller computed dimension
        computed_tile_width = base_width // num_cols
        computed_tile_height = base_height // num_rows
        tile_size = min(computed_tile_width, computed_tile_height)
    else:
        # Default to computed (non-square) sizes if nothing is specified.
        tile_size = None

    if tile_size is not None:
        # Enforce that all tiles are square and set mosaic dimensions accordingly.
        mosaic_width = num_cols * tile_size
        mosaic_height = num_rows * tile_size

        # Resize the base image to exactly match the mosaic dimensions.
        base_image = base_image.resize((mosaic_width, mosaic_height), resample=Image.LANCZOS)
    else:
        # If no tile_size is specified, fall back to computed dimensions.
        mosaic_width = base_width
        mosaic_height = base_height
        # Compute non-square tile dimensions
        computed_tile_width = base_width // num_cols
        computed_tile_height = base_height // num_rows

    # Create mosaic with margin
    final_width = mosaic_width + (2 * margin)
    final_height = mosaic_height + (2 * margin)
    mosaic = Image.new("RGB", (final_width, final_height), "white")

    # Create a pixelated version of the base image
    # First, resize to a small size to create the pixelation effect
    small_size = (num_cols, num_rows)
    pixelated = base_image.resize(small_size, resample=Image.NEAREST)
    # Then resize back to the original size
    pixelated = pixelated.resize((mosaic_width, mosaic_height), resample=Image.NEAREST)

    # Apply blur to the pixelated image
    pixelated = pixelated.filter(ImageFilter.GaussianBlur(radius=5))

    # Apply background opacity if less than 100%
    if bg_opacity < 100:
        pixelated = pixelated.convert("RGBA")
        bg_image = Image.new("RGBA", pixelated.size, (255, 255, 255, 0))
        mask = Image.new("L", pixelated.size, int(bg_opacity * 2.55))
        pixelated = Image.composite(pixelated, bg_image, mask)

    # Paste the pixelated background onto the mosaic
    mosaic.paste(pixelated, (margin, margin))

    # Now overlay the QR codes
    link_index = 0
    for row in range(num_rows):
        for col in range(num_cols):
            if tile_size is not None:
                current_tile_width = tile_size
                current_tile_height = tile_size
            else:
                current_tile_width = computed_tile_width
                current_tile_height = computed_tile_height

            # Get the URL for this tile; use a fallback if needed.
            if link_index < len(df):
                url = df.iloc[link_index]["URL"]
            else:
                url = "https://example.com"
            link_index += 1

            # Generate the QR code.
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=0,
            )
            qr.add_data(url)
            qr.make(fit=True)
            qr_img = qr.make_image(fill_color="black", back_color="white")

            # Resize QR code to fit within the tile with padding
            qr_size = min(current_tile_width, current_tile_height) - (2 * qr_padding)
            qr_img = qr_img.resize((qr_size, qr_size), resample=Image.LANCZOS)

            # Calculate position to center the QR code in the tile
            paste_x = margin + (col * current_tile_width) + (current_tile_width - qr_size) // 2
            paste_y = margin + (row * current_tile_height) + (current_tile_height - qr_size) // 2

            # Apply QR code opacity if less than 100%
            if qr_opacity < 100:
                qr_img = qr_img.convert("RGBA")
                # Create a new image with the same size
                qr_bg = Image.new("RGBA", qr_img.size, (255, 255, 255, 0))
                # Create a mask with the desired opacity
                qr_mask = Image.new("L", qr_img.size, int(qr_opacity * 2.55))
                # Composite the QR code with the mask
                qr_img = Image.composite(qr_img, qr_bg, qr_mask)

            # Paste the QR code onto the mosaic
            mosaic.paste(qr_img, (paste_x, paste_y), qr_img)

    # Save the final image
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
    mosaic.save(result_path, 'JPEG', quality=95, dpi=(dpi, dpi))
    return result_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Add detailed debug prints
        print("Request method:", request.method)
        print("Request files:", request.files)
        print("Request form:", request.form)
        print("Files in request:", list(request.files.keys()))
        print("Image file present:", 'image' in request.files)
        print("Excel file present:", 'excel' in request.files)
        
        if 'image' not in request.files or 'excel' not in request.files:
            error_msg = f"Missing files. Image: {'image' in request.files}, Excel: {'excel' in request.files}"
            print(error_msg)
            return render_template('index.html', error=error_msg)
        
        image_file = request.files['image']
        excel_file = request.files['excel']
        
        if image_file.filename == '' or excel_file.filename == '':
            return render_template('index.html', error='No selected files')
        
        if not (allowed_file(image_file.filename) and allowed_file(excel_file.filename)):
            return render_template('index.html', error='Invalid file type')
        
        # Clean up old files
        cleanup_old_files()
        
        # Generate unique filenames
        image_filename = get_unique_filename(image_file.filename)
        excel_filename = get_unique_filename(excel_file.filename)
        
        # Save uploaded files
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        excel_path = os.path.join(app.config['UPLOAD_FOLDER'], excel_filename)
        
        image_file.save(image_path)
        excel_file.save(excel_path)
        
        # Get form parameters
        num_cols = int(request.form.get('num_cols', 20))
        num_rows = int(request.form.get('num_rows', 20))
        square_tiles = 'square_tiles' in request.form
        tile_size = request.form.get('tile_size_pixels')
        tile_size = int(tile_size) if tile_size and tile_size.isdigit() else None
        
        try:
            # Get pixel values instead of direct inputs
            margin = request.form.get('margin_pixels')
            margin = int(margin) if margin and margin.isdigit() else 0
            
            qr_padding = request.form.get('qr_padding_pixels')
            qr_padding = int(qr_padding) if qr_padding and qr_padding.isdigit() else 0
            
            # Get QR opacity parameter
            qr_opacity = request.form.get('qr_opacity', '100')
            qr_opacity = int(qr_opacity) if qr_opacity.isdigit() else 100
            qr_opacity = max(0, min(100, qr_opacity))  # Clamp between 0 and 100
            
            # Get background opacity parameter
            bg_opacity = request.form.get('bg_opacity', '100')
            bg_opacity = int(bg_opacity) if bg_opacity.isdigit() else 100
            bg_opacity = max(0, min(100, bg_opacity))  # Clamp between 0 and 100
            
            # Get size parameters
            width_inches = request.form.get('width_inches', '0')
            width_inches = float(width_inches) if width_inches else 0
            
            height_inches = request.form.get('height_inches', '0')
            height_inches = float(height_inches) if height_inches else 0
            
            dpi = request.form.get('dpi', '300')
            dpi = int(dpi) if dpi.isdigit() else 300
            dpi = max(72, min(1200, dpi))  # Clamp between 72 and 1200
            
            result_path = generate_qr_mosaic(
                image_path,
                excel_path,
                num_cols,
                num_rows,
                square_tiles,
                tile_size,
                margin,
                qr_padding,
                qr_opacity,
                width_inches,
                height_inches,
                dpi,
                bg_opacity
            )
            return render_template('index.html', result=True)
        except Exception as e:
            return render_template('index.html', error=str(e))
    
    return render_template('index.html')

@app.route('/download')
def download():
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg'),
                    as_attachment=True,
                    download_name='qr_mosaic.jpg')

@app.route('/health')
def health_check():
    return {'status': 'healthy'}, 200

@app.route('/preview', methods=['POST'])
def preview():
    if 'image' not in request.files or 'excel' not in request.files:
        return jsonify({'error': 'Missing files'}), 400
    
    image_file = request.files['image']
    excel_file = request.files['excel']
    
    if image_file.filename == '' or excel_file.filename == '':
        return jsonify({'error': 'No selected files'}), 400
    
    if not (allowed_file(image_file.filename) and allowed_file(excel_file.filename)):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Clean up old files
    cleanup_old_files()
    
    # Generate unique filenames
    image_filename = get_unique_filename(image_file.filename)
    excel_filename = get_unique_filename(excel_file.filename)
    
    # Save uploaded files
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    excel_path = os.path.join(app.config['UPLOAD_FOLDER'], excel_filename)
    
    image_file.save(image_path)
    excel_file.save(excel_path)
    
    try:
        # Get form parameters
        num_cols = int(request.form.get('num_cols', 20))
        num_rows = int(request.form.get('num_rows', 20))
        square_tiles = 'square_tiles' in request.form
        tile_size = request.form.get('tile_size_pixels')
        tile_size = int(tile_size) if tile_size and tile_size.isdigit() else None
        
        margin = request.form.get('margin_pixels')
        margin = int(margin) if margin and margin.isdigit() else 0
        
        qr_padding = request.form.get('qr_padding_pixels')
        qr_padding = int(qr_padding) if qr_padding and qr_padding.isdigit() else 0
        
        qr_opacity = request.form.get('qr_opacity', '100')
        qr_opacity = int(qr_opacity) if qr_opacity.isdigit() else 100
        qr_opacity = max(0, min(100, qr_opacity))
        
        bg_opacity = request.form.get('bg_opacity', '100')
        bg_opacity = int(bg_opacity) if bg_opacity.isdigit() else 100
        bg_opacity = max(0, min(100, bg_opacity))
        
        width_inches = request.form.get('width_inches', '0')
        width_inches = float(width_inches) if width_inches else 0
        
        height_inches = request.form.get('height_inches', '0')
        height_inches = float(height_inches) if height_inches else 0
        
        dpi = request.form.get('dpi', '300')
        dpi = int(dpi) if dpi.isdigit() else 300
        dpi = max(72, min(1200, dpi))
        
        # Check memory requirements for the full-size image
        memory_validation = validate_memory_requirements(width_inches, height_inches, dpi, margin)
        
        result_path = generate_qr_mosaic(
            image_path,
            excel_path,
            num_cols,
            num_rows,
            square_tiles,
            tile_size,
            margin,
            qr_padding,
            qr_opacity,
            width_inches,
            height_inches,
            dpi,
            bg_opacity,
            preview=True
        )
        
        # Convert the image to base64 for the preview
        with open(result_path, 'rb') as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        return jsonify({
            'preview': f'data:image/jpeg;base64,{img_base64}',
            'memory_info': memory_validation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True) 