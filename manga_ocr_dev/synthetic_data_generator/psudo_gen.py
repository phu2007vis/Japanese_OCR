import os
import csv
from PIL import Image, ImageDraw, ImageFont
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from manga_ocr_dev.env import DATA_SYNTHETIC_ROOT
# Configuration
root_dir =  DATA_SYNTHETIC_ROOT
output_dir = os.path.join(root_dir,"img","0000")

csv_filename = os.path.join(root_dir,r"meta\0000.csv")
font_path = "NotoSansJP-Medium.ttf"
text = "発展を"
vertical = True
source = "cc-100"
image_size = (512, 512)
font_size = 48
num_images = 200

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# # Load font
# try:
#     font = ImageFont.truetype(font_path, font_size)
# except IOError:
#     print(f"Font file '{font_path}' not found.")
#     exit(1)

# Prepare CSV file
with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["source", "id", "text", "vertical", "font_path"])

    for i in range(num_images):
        # Generate unique ID
        unique_id = f"{source}_{446088 + i}"
        image_filename = f"{unique_id}.png"
        image_path = os.path.join(output_dir, image_filename)

        # Create a new image with white background
        img = Image.new("RGB", image_size, color="white")
        draw = ImageDraw.Draw(img)

  
        # Render vertical text
        chars = list(text)
        total_height = 0
        char_images = []

        # Save image
        img.save(image_path)

        # Write metadata to CSV
        writer.writerow([source, unique_id, text, vertical, font_path])
