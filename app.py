import os
from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import pandas as pd
import qrcode
import base64
import psutil
import math
import tempfile
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'static/uploads')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'xlsx', 'xls'}

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
    
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Process the image in chunks if it's very large
        chunk_size = 1000  # Process 1000 rows at a time
        final_image = Image.new('RGB', (base_width + 2 * margin, base_height + 2 * margin), 'white')
        
        for row_start in range(0, num_rows, chunk_size):
            row_end = min(row_start + chunk_size, num_rows)
            
            # Create a chunk of the final image
            chunk_height = int((row_end - row_start) * base_height / num_rows)
            chunk_image = Image.new('RGB', (base_width + 2 * margin, chunk_height), 'white')
            
            # Process this chunk
            for row in range(row_start, row_end):
                for col in range(num_cols):
                    if tile_size is not None:
                        current_tile_width = tile_size
                        current_tile_height = tile_size
                    else:
                        current_tile_width = base_width // num_cols
                        current_tile_height = base_height // num_rows

                    # Calculate crop coordinates for the current tile from the resized base image.
                    x0 = col * current_tile_width
                    y0 = row * current_tile_height
                    x1 = x0 + current_tile_width
                    y1 = y0 + current_tile_height
                    tile_region = base_image.crop((x0, y0, x1, y1))

                    # Compute the average color for tinting.
                    avg_color = tile_region.resize((1, 1)).getpixel((0, 0))

                    # Get the URL for this tile; use a fallback if needed.
                    if row < len(df):
                        url = df.iloc[row]["URL"]
                    else:
                        url = "https://example.com"

                    # Generate the QR code.
                    qr = qrcode.QRCode(
                        version=1,
                        error_correction=qrcode.constants.ERROR_CORRECT_H,
                        box_size=10,
                        border=1
                    )
                    qr.add_data(url)
                    qr.make(fit=True)

                    # Create the QR code image and convert to RGBA for opacity support
                    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGBA")

                    # Calculate the size for QR code with padding
                    padded_width = current_tile_width - (2 * qr_padding)
                    padded_height = current_tile_height - (2 * qr_padding)
                    
                    # Ensure minimum size
                    padded_width = max(padded_width, 10)
                    padded_height = max(padded_height, 10)

                    # Resize the QR code to the padded size
                    qr_img = qr_img.resize((padded_width, padded_height), resample=Image.LANCZOS)

                    # Create a new RGBA image for the tile with padding
                    tile_img = Image.new("RGBA", (current_tile_width, current_tile_height), (255, 255, 255, 0))

                    # Calculate position to center the QR code in the tile
                    paste_x = qr_padding
                    paste_y = qr_padding

                    # Apply color tinting and opacity
                    pixels = qr_img.load()
                    width_qr, height_qr = qr_img.size
                    for px in range(width_qr):
                        for py in range(height_qr):
                            r, g, b, a = pixels[px, py]
                            if (r, g, b) == (0, 0, 0):  # Dark modules
                                dr = int(avg_color[0] * 0.5)
                                dg = int(avg_color[1] * 0.5)
                                db = int(avg_color[2] * 0.5)
                                opacity = int((qr_opacity / 100.0) * 255)
                                pixels[px, py] = (dr, dg, db, opacity)
                            else:  # Light modules
                                lr = int((avg_color[0] + 255) / 2)
                                lg = int((avg_color[1] + 255) / 2)
                                lb = int((avg_color[2] + 255) / 2)
                                opacity = int((qr_opacity / 100.0) * 255)
                                pixels[px, py] = (lr, lg, lb, opacity)

                    # Paste the QR code onto the padded tile
                    tile_img.paste(qr_img, (paste_x, paste_y))

                    # Paste the tile onto the chunk image
                    chunk_image.paste(tile_img, (margin + (col * current_tile_width), margin + (row * current_tile_height)))
            
            # Paste the chunk into the final image
            y_offset = int(row_start * base_height / num_rows) + margin
            final_image.paste(chunk_image, (margin, y_offset))
            
            # Clear memory
            del chunk_image
        
        # Save the final image
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
        final_image.save(result_path, 'JPEG', quality=95)
        
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
        
        # Save uploaded files
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
        excel_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(excel_file.filename))
        
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
    
    # Save uploaded files
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
    excel_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(excel_file.filename))
    
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