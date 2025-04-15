from pathlib import Path
import cv2
import numpy as np
import random
from tqdm import tqdm
import os 
import sys

# Define your own image folder and output folder here
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from manga_ocr_dev.env import ORIGINAL_BACKGROUND, BACKGROUND_DIR
IMAGE_DIR = Path(ORIGINAL_BACKGROUND)  # Replace with your folder of comic/manga images
BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)
print(IMAGE_DIR)

def find_rectangle(mask, y, x, aspect_ratio_range=(0.33, 3.0)):
    ymin_ = ymax_ = y
    xmin_ = xmax_ = x
    ymin = ymax = xmin = xmax = None

    while True:
        if ymin is None:
            ymin_ -= 1
            if ymin_ <= 0 or mask[ymin_, xmin_:xmax_].any():
                ymin = ymin_

        if ymax is None:
            ymax_ += 1
            if ymax_ >= mask.shape[0] - 1 or mask[ymax_, xmin_:xmax_].any():
                ymax = ymax_

        if xmin is None:
            xmin_ -= 1
            if xmin_ <= 0 or mask[ymin_:ymax_, xmin_].any():
                xmin = xmin_

        if xmax is None:
            xmax_ += 1
            if xmax_ >= mask.shape[1] - 1 or mask[ymin_:ymax_, xmax_].any():
                xmax = xmax_

        h = ymax_ - ymin_
        w = xmax_ - xmin_
        if h > 1 and w > 1:
            ratio = w / h
            if ratio < aspect_ratio_range[0] or ratio > aspect_ratio_range[1]:
                return ymin_, ymax_, xmin_, xmax_

        if None not in (ymin, ymax, xmin, xmax):
            return ymin, ymax, xmin, xmax


def generate_backgrounds_from_images(image_dir, crops_per_image=5, min_size=40):
    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

    for img_path in tqdm(image_paths):
        page = cv2.imread(str(img_path))
        if page is None:
            print(f"Could not read {img_path}")
            continue

        # Simulated mask: randomly block out some areas (pretend they are text regions)
        mask = np.zeros((page.shape[0], page.shape[1]), dtype=bool)
        for _ in range(10):  # Simulate some 'masked' boxes
            h, w = page.shape[:2]
            x1 = random.randint(0, w - 50)
            y1 = random.randint(0, h - 50)
            x2 = min(w, x1 + random.randint(30, 150))
            y2 = min(h, y1 + random.randint(30, 150))
            mask[y1:y2, x1:x2] = True

        if mask.all():
            continue

        unmasked_points = np.stack(np.where(~mask), axis=1)
        for _ in range(crops_per_image):
            p = unmasked_points[np.random.randint(0, unmasked_points.shape[0])]
            y, x = p
            ymin, ymax, xmin, xmax = find_rectangle(mask, y, x)
            crop = page[ymin:ymax, xmin:xmax]

            if crop.shape[0] >= min_size and crop.shape[1] >= min_size:
                out_filename = (
                    f"{img_path.stem}_{ymin}_{ymax}_{xmin}_{xmax}.png"
                )
                cv2.imwrite(str(BACKGROUND_DIR / out_filename), crop)


if __name__ == "__main__":
    generate_backgrounds_from_images(IMAGE_DIR)
