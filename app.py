import os
from flask import Flask, render_template, request, send_file, jsonify, session
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter, ImageDraw
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
from colorsys import rgb_to_hls, hls_to_rgb
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import (
    SquareModuleDrawer,
    CircleModuleDrawer,
    RoundedModuleDrawer,
    GappedSquareModuleDrawer,
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'static/uploads')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')  # Change this in production
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)  # Session expires after 1 hour

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

def get_luminance(color):
    r, g, b = color
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def adjust_color_lighter(color, factor=1.2):
    # Get the luminance of the original color
    luminance = get_luminance(color)
    
    # Adjust the factor based on luminance
    # Darker colors get a higher factor to make them lighter
    if luminance < 128:  # If color is dark
        adjusted_factor = factor * (1 + (128 - luminance) / 128)
    else:  # If color is light
        adjusted_factor = factor
    
    return tuple(max(0, min(255, int(c * adjusted_factor))) for c in color)

def adjust_saturation(color, saturation=1.0):
    # color: (r, g, b), values 0-255
    r, g, b = [c / 255.0 for c in color]
    h, l, s = rgb_to_hls(r, g, b)
    s = max(0, min(1, s * saturation))
    r, g, b = hls_to_rgb(h, l, s)
    return (int(r * 255), int(g * 255), int(b * 255))

def get_qr_style(style_name):
    """Get the QR code module drawer based on the selected style"""
    styles = {
        'default': SquareModuleDrawer(),  # Use SquareModuleDrawer for default style
        'rounded': RoundedModuleDrawer(),
        'circle': CircleModuleDrawer(),
        'square': SquareModuleDrawer(),
        'gapped': GappedSquareModuleDrawer(),
    }
    return styles.get(style_name, SquareModuleDrawer())  # Default to SquareModuleDrawer if style not found

def generate_qr_mosaic(image_path, excel_path, num_cols, num_rows, square_tiles, tile_size, 
                      margin=0, qr_opacity=100, width_inches=0, height_inches=0, dpi=300, bg_opacity=100, preview=False, download_type='jpg', tile_gap=0, qr_shade=1.2, qr_saturation=1.0):
    # Load Excel data
    df = pd.read_excel(excel_path)
    
    # Convert decimal values to integers for pixel calculations
    margin = int(margin)
    if tile_size is not None:
        tile_size = int(tile_size)
    # Convert tile_gap from inches to pixels
    gap = int(float(tile_gap) * dpi) if tile_gap else 0
    
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

    # Create a pixelated version of the base image for color sampling (before opacity)
    pixelated_for_color = base_image.resize((num_cols, num_rows), resample=Image.NEAREST)
    pixelated_for_color = pixelated_for_color.resize((mosaic_width, mosaic_height), resample=Image.NEAREST)
    pixelated = pixelated_for_color.copy()
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
    if not gap or gap < 2:
        gap = 2
    for row in range(num_rows):
        for col in range(num_cols):
            if tile_size is not None:
                current_tile_width = tile_size
                current_tile_height = tile_size
            else:
                current_tile_width = computed_tile_width
                current_tile_height = computed_tile_height

            # Get the URL for this tile; repeat the list if needed.
            url = df.iloc[link_index % len(df)]["URL"]
            link_index += 1

            tile_x = col * current_tile_width
            tile_y = row * current_tile_height
            # Always sample color from the original pixelated image (before opacity)
            region_for_color = pixelated_for_color.crop((tile_x, tile_y, tile_x + current_tile_width, tile_y + current_tile_height))
            avg_color = region_for_color.resize((1, 1), resample=Image.LANCZOS).getpixel((0, 0))

            # Use the average color for the QR code, white for the background
            qr_color = adjust_saturation(adjust_color_lighter(avg_color, 1), 1)
            bg_color = (255, 255, 255)

            # Generate QR code with the selected colors
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=0,
            )
            qr.add_data(url)
            qr.make(fit=True)
            qr_img = qr.make_image(fill_color=qr_color, back_color=bg_color).convert('RGBA')

            # Resize QR code to fit within the tile with the gap
            qr_size = min(current_tile_width, current_tile_height) - (2 * gap)
            qr_size = max(qr_size, 1)
            qr_img = qr_img.resize((qr_size, qr_size), resample=Image.LANCZOS)

            # Apply QR code opacity if less than 100%
            if qr_opacity < 100:
                alpha = qr_img.split()[3].point(lambda p: int(p * (qr_opacity / 100)))
                qr_img.putalpha(alpha)

            # Calculate position to center the QR code in the tile (respecting gap)
            paste_x = margin + (col * current_tile_width) + gap + (current_tile_width - 2 * gap - qr_size) // 2
            paste_y = margin + (row * current_tile_height) + gap + (current_tile_height - 2 * gap - qr_size) // 2

            # Paste the QR code onto the mosaic
            mosaic.paste(qr_img, (paste_x, paste_y), qr_img)

    # Save the final image
    if download_type == 'png':
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.png')
        mosaic.save(result_path, 'PNG', dpi=(dpi, dpi))
    elif download_type == 'pdf':
        # For PDF, save as PNG first
        temp_png = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_result.png')
        mosaic.save(temp_png, 'PNG', dpi=(dpi, dpi))
        
        # Convert PNG to PDF
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.pdf')
        img = Image.open(temp_png)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(result_path, 'PDF', resolution=dpi)
        
        # Clean up temporary PNG file
        try:
            os.remove(temp_png)
        except:
            pass
    else:
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
        print("Current session data:", dict(session))  # Debug print
        
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
        
        try:
            # Get form parameters
            num_cols = int(request.form.get('num_cols', 20))
            num_rows = int(request.form.get('num_rows', 20))
            square_tiles = 'square_tiles' in request.form
            tile_size = request.form.get('tile_size_pixels')
            tile_size = float(tile_size) if tile_size and tile_size.replace('.', '').isdigit() else None
            
            # Get pixel values instead of direct inputs
            margin = request.form.get('margin_pixels')
            margin = float(margin) if margin and margin.replace('.', '').isdigit() else 0
            
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
            
            download_type = request.form.get('download_type', 'jpg')
            
            # Get tile_gap parameter
            tile_gap = request.form.get('tile_gap', '0')
            tile_gap = float(tile_gap) if tile_gap else 0
            
            # Get qr_shade parameter
            qr_shade = float(request.form.get('qr_shade', '0.8'))
            
            # Get qr_saturation parameter
            qr_saturation = float(request.form.get('qr_saturation', '1.0'))
            
            print("Generating mosaic with parameters:")
            print(f"num_cols: {num_cols}, num_rows: {num_rows}")
            print(f"tile_size: {tile_size}, margin: {margin}, qr_opacity: {qr_opacity}")
            print(f"width_inches: {width_inches}, height_inches: {height_inches}, dpi: {dpi}")
            
            result_path = generate_qr_mosaic(
                image_path,
                excel_path,
                num_cols,
                num_rows,
                square_tiles,
                tile_size,
                margin,
                qr_opacity,
                width_inches,
                height_inches,
                dpi,
                bg_opacity,
                download_type='png',  # Always generate PNG for storage
                tile_gap=tile_gap,
                qr_shade=qr_shade,
                qr_saturation=qr_saturation
            )
            
            # Store the result path in session
            session['result_path'] = result_path
            session.permanent = True  # Make the session persistent
            
            print(f"Mosaic generated successfully. Result path: {result_path}")
            print(f"Session data after generation: {dict(session)}")  # Debug print
            
            return render_template('index.html', result=True)
        except Exception as e:
            print(f"Error generating mosaic: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return render_template('index.html', error=str(e))
    
    return render_template('index.html')

@app.route('/preview', methods=['POST'])
def preview():
    try:
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
            tile_size = float(tile_size) if tile_size and tile_size.replace('.', '').isdigit() else None
            
            margin = request.form.get('margin_pixels')
            margin = float(margin) if margin and margin.replace('.', '').isdigit() else 0
            
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
            
            # For preview, always use PNG format
            download_type = 'png'
            
            # Get tile_gap parameter
            tile_gap = request.form.get('tile_gap', '0')
            tile_gap = float(tile_gap) if tile_gap else 0
            
            # Get qr_shade parameter
            qr_shade = float(request.form.get('qr_shade', '0.8'))
            
            # Get qr_saturation parameter
            qr_saturation = float(request.form.get('qr_saturation', '1.0'))
            
            print("Generating preview with parameters:")
            print(f"num_cols: {num_cols}, num_rows: {num_rows}")
            print(f"tile_size: {tile_size}, margin: {margin}, qr_opacity: {qr_opacity}")
            print(f"width_inches: {width_inches}, height_inches: {height_inches}, dpi: {dpi}")
            
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
                qr_opacity,
                width_inches,
                height_inches,
                dpi,
                bg_opacity,
                preview=True,
                download_type=download_type,
                tile_gap=tile_gap,
                qr_shade=qr_shade,
                qr_saturation=qr_saturation
            )
            
            # For preview, always return a PNG base64 image
            with open(result_path, 'rb') as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            return jsonify({
                'preview': f'data:image/png;base64,{img_base64}',
                'memory_info': memory_validation
            })
        except Exception as e:
            print(f"Error in preview generation: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    except Exception as e:
        print(f"Error in preview endpoint: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/download', methods=['POST'])
def download():
    result_path = session.get('result_path')
    download_type = request.form.get('download_type', 'jpg')
    
    print(f"Download request - Session data: {dict(session)}")  # Debug print
    print(f"Download type requested: {download_type}")  # Debug print
    print(f"Result path from session: {result_path}")  # Debug print
    
    if not result_path:
        print("No result_path in session")  # Debug print
        return "No file path found in session. Please generate the mosaic again.", 404
    
    if not os.path.exists(result_path):
        print(f"File not found at path: {result_path}")  # Debug print
        return "Generated file not found. Please generate the mosaic again.", 404
    
    try:
        # Get the file extension based on download type
        if download_type == 'pdf':
            # For PDF, we need to ensure it's a valid PDF file
            if not result_path.endswith('.pdf'):
                # Convert the image to PDF
                img = Image.open(result_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.pdf')
                img.save(pdf_path, 'PDF', resolution=300)
                result_path = pdf_path
            
            return send_file(
                result_path,
                as_attachment=True,
                download_name='qr_mosaic.pdf',
                mimetype='application/pdf'
            )
        elif download_type == 'png':
            return send_file(
                result_path,
                as_attachment=True,
                download_name='qr_mosaic.png',
                mimetype='image/png'
            )
        else:  # jpg
            return send_file(
                result_path,
                as_attachment=True,
                download_name='qr_mosaic.jpg',
                mimetype='image/jpeg'
            )
    except Exception as e:
        print(f"Error in download: {str(e)}")
        import traceback
        print(traceback.format_exc())  # Print full traceback for debugging
        return f"Error downloading file: {str(e)}", 500

@app.route('/health')
def health_check():
    return {'status': 'healthy'}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True) 