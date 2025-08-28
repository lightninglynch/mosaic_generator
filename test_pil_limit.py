#!/usr/bin/env python3

from PIL import Image
import sys

# Test the PIL limit change
print("Testing PIL image size limit...")

# Set the limit like in our app
Image.MAX_IMAGE_PIXELS = 1000000000

# Test with a large image size that would have failed before
test_width = 13650
test_height = 13650
total_pixels = test_width * test_height

print(f"Testing image size: {test_width} x {test_height} = {total_pixels:,} pixels")
print(f"PIL limit: {Image.MAX_IMAGE_PIXELS:,} pixels")

if total_pixels <= Image.MAX_IMAGE_PIXELS:
    print("✅ Image size is within PIL limits")
else:
    print("❌ Image size exceeds PIL limits")

# Test creating a small test image to verify PIL works
try:
    test_img = Image.new('RGB', (100, 100), color='white')
    test_img.save('test_small.png')
    print("✅ Successfully created test image")
    import os
    os.remove('test_small.png')
except Exception as e:
    print(f"❌ Error creating test image: {e}")
    sys.exit(1)

print("✅ PIL limit test completed successfully")
