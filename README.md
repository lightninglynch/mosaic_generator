# QR Mosaic Generator

A web application that generates QR code mosaics from images and Excel spreadsheets. Each QR code in the mosaic links to a different URL from your Excel file and is color-matched to the corresponding region of the original image.

## Features

- Upload any image to create a QR code mosaic
- Upload an Excel file containing URLs to be encoded in the QR codes
- Customize mosaic dimensions and tile sizes
- Adjust QR code appearance (opacity, padding, margins)
- Set output image dimensions in inches with DPI control
- Download the generated mosaic

## Requirements

- Python 3.x
- Flask
- Pillow (PIL)
- pandas
- qrcode
- openpyxl
- gunicorn (for production deployment)

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd mosaic_generator
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Upload your files:
   - An image file (JPG, PNG)
   - An Excel file containing a "URL" column with the links you want to encode

4. Configure the mosaic settings:
   - Number of columns and rows
   - Tile size (optional)
   - Square tiles option
   - Margin and padding
   - QR code opacity
   - Output dimensions (in inches)
   - DPI settings

5. Click "Generate Mosaic" to create your QR mosaic

6. Download the generated mosaic using the download button

## Excel File Format

Your Excel file should contain a column named "URL" with the links you want to encode in the QR codes. The application will use these URLs in sequence to create the mosaic.

Example Excel format:
```
URL
https://example.com/1
https://example.com/2
https://example.com/3
...
```

## Deployment

For production deployment, you can use the provided `Dockerfile` and `deploy.sh` script:

```bash
./deploy.sh
```

## License

[Your License Here]

## Contributing

[Your Contribution Guidelines Here] 