import os
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import pandas as pd
import qrcode

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'static/uploads')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'xlsx', 'xls'}

def generate_qr_mosaic(image_path, excel_path, num_cols, num_rows, square_tiles, tile_size, 
                      margin=0, qr_padding=0, qr_opacity=100, width_inches=0, height_inches=0, dpi=300):
    # Load Excel data
    df = pd.read_excel(excel_path)
    
    # Load the base image
    base_image = Image.open(image_path)
    base_width, base_height = base_image.size

    # If inches are specified, convert to pixels using DPI
    if width_inches > 0 and height_inches > 0:
        target_width = int(width_inches * dpi)
        target_height = int(height_inches * dpi)
        
        # Resize base image to match target dimensions
        base_image = base_image.resize((target_width, target_height), resample=Image.LANCZOS)
        base_width, base_height = base_image.size

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

    link_index = 0
    for row in range(num_rows):
        for col in range(num_cols):
            if tile_size is not None:
                current_tile_width = tile_size
                current_tile_height = tile_size
            else:
                current_tile_width = computed_tile_width
                current_tile_height = computed_tile_height

            # Calculate crop coordinates for the current tile from the resized base image.
            x0 = col * current_tile_width
            y0 = row * current_tile_height
            x1 = x0 + current_tile_width
            y1 = y0 + current_tile_height
            tile_region = base_image.crop((x0, y0, x1, y1))

            # Compute the average color for tinting.
            avg_color = tile_region.resize((1, 1)).getpixel((0, 0))

            # Get the URL for this tile; use a fallback if needed.
            if link_index < len(df):
                url = df.iloc[link_index]["URL"]
            else:
                url = "https://example.com"
            link_index += 1

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

            # Paste the tile onto the mosaic
            mosaic_x = margin + (col * current_tile_width)
            mosaic_y = margin + (row * current_tile_height)
            
            # Composite the tile with the base image
            mosaic.paste(tile_img, (mosaic_x, mosaic_y), tile_img)

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_mosaic.jpg')
    # Convert back to RGB before saving as JPG
    mosaic = mosaic.convert('RGB')
    mosaic.save(output_path, dpi=(dpi, dpi))  # Save with specified DPI
    return output_path

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
                dpi
            )
            return render_template('index.html', result=True)
        except Exception as e:
            return render_template('index.html', error=str(e))
    
    return render_template('index.html')

@app.route('/download')
def download():
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], 'result_mosaic.jpg'),
                    as_attachment=True,
                    download_name='qr_mosaic.jpg')

@app.route('/health')
def health_check():
    return {'status': 'healthy'}, 200

if __name__ == '__main__':
    app.run(debug=True) 