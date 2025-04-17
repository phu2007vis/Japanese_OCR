import json
from pathlib import Path
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from manga_ocr import MangaOcr
import cv2
import shutil
TEST_DATA_ROOT = Path(__file__).parent / "data2"
save_folder = Path(__file__).parent / "data3"

if os.path.exists(save_folder):
    import shutil
    shutil.rmtree(save_folder)
    
save_folder.mkdir(exist_ok=True)
def test_ocr():
	mocr = MangaOcr(pretrained_model_name_or_path = "/work/21013187/phuoc/Japanese_OCR/weights_main/phuoc")

	for file_name in os.listdir(TEST_DATA_ROOT / "images"):
		path_img = TEST_DATA_ROOT / "images" / file_name
		result = mocr(path_img)
		shutil.copy(path_img,os.path.join(save_folder,f"{result}{file_name}"))

if __name__ == "__main__":
	test_ocr()