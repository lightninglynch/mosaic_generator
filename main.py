import argparse
import pandas as pd
from PIL import Image, ImageFilter
import qrcode

def main():
    parser = argparse.ArgumentParser(description="QR mosaic generator")
    parser.add_argument("--num_cols", type=int, default=20, help="Number of columns in the mosaic grid")
    parser.add_argument("--num_rows", type=int, default=20, help="Number of rows in the mosaic grid")
    parser.add_argument("--square_tiles", action="store_true", help="Force each tile to be square")
    parser.add_argument("--tile_size", type=int, default=None, help="Specify tile size in pixels (square) to override computed tile size")
    parser.add_argument("--input_excel", type=str, default="links.xlsx", help="Path to Excel file with URL column")
    parser.add_argument("--input_image", type=str, default="portrait.jpg", help="Path to base image")
    parser.add_argument("--output_image", type=str, default="qr_mosaic.jpg", help="Path to output mosaic image")
    args = parser.parse_args()

    # Load Excel data (must have a "URL" column)
    df = pd.read_excel(args.input_excel)

    # Load the base image
    base_image = Image.open(args.input_image)
    base_width, base_height = base_image.size

    # Determine the tile size
    if args.tile_size:
        # Use the tile size provided by the user
        tile_size = args.tile_size
    elif args.square_tiles:
        # Force square tiles using the smaller computed dimension
        computed_tile_width = base_width // args.num_cols
        computed_tile_height = base_height // args.num_rows
        tile_size = min(computed_tile_width, computed_tile_height)
    else:
        # Default to computed (non-square) sizes if nothing is specified.
        tile_size = None

    if tile_size is not None:
        # Enforce that all tiles are square and set mosaic dimensions accordingly.
        mosaic_width = args.num_cols * tile_size
        mosaic_height = args.num_rows * tile_size

        # Resize the base image to exactly match the mosaic dimensions.
        base_image = base_image.resize((mosaic_width, mosaic_height), resample=Image.LANCZOS)
    else:
        # If no tile_size is specified, fall back to computed dimensions.
        mosaic_width = base_width
        mosaic_height = base_height
        # Compute non-square tile dimensions (though this code path won't force same-sized square QR codes)
        computed_tile_width = base_width // args.num_cols
        computed_tile_height = base_height // args.num_rows

    mosaic = Image.new("RGB", (mosaic_width, mosaic_height), "white")

    link_index = 0
    for row in range(args.num_rows):
        for col in range(args.num_cols):
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
                error_correction=qrcode.constants.ERROR_CORRECT_H,  # Allows color modifications
                box_size=10,
                border=1
            )
            qr.add_data(url)
            qr.make(fit=True)

            # Create the QR code image (default black/white) and convert to RGB.
            qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

            # Recolor the QR code: tint black modules with a darker version of the tile's average color,
            # and adjust white areas to a lighter tone.
            pixels = qr_img.load()
            width_qr, height_qr = qr_img.size
            for px in range(width_qr):
                for py in range(height_qr):
                    r, g, b = pixels[px, py]
                    if (r, g, b) == (0, 0, 0):
                        dr = int(avg_color[0] * 0.5)
                        dg = int(avg_color[1] * 0.5)
                        db = int(avg_color[2] * 0.5)
                        pixels[px, py] = (dr, dg, db)
                    else:
                        lr = int((avg_color[0] + 255) / 2)
                        lg = int((avg_color[1] + 255) / 2)
                        lb = int((avg_color[2] + 255) / 2)
                        pixels[px, py] = (lr, lg, lb)

            # Resize the QR code to the fixed (square) tile size.
            qr_img = qr_img.resize((current_tile_width, current_tile_height), resample=Image.LANCZOS)

            # Paste the QR code into the mosaic.
            mosaic.paste(qr_img, (x0, y0))

    mosaic.save(args.output_image)
    print("Mosaic saved to", args.output_image)

if __name__ == "__main__":
    main()
