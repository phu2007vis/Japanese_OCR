# TODO

1. Scrip for text generation
2. New pretrained weights
3. If you found this repository helpful or inspiring, please consider starring it.
4. Your support encourages further development and enhancements

# Test 
![plot](data/visualize.png)
# Some sample synthetic data
![plot](data/595.png)
# Key requirements
- transformers >= 4.5x 
- python 3.11 is recommended which is my case (i also test run well with python 3.9)
# Download weight 
```bash
wget https://github.com/phu2007vis/Japanese_OCR/releases/download/weights/weights_main.zip
unzip weights_main
```
```
Ensure the stucture of weight is:
--root
    --weights_main
        --phuoc
            --all file weights and config here
```
Can also customize load weight process at manga_ocr_dev/training/get_model.py

# Fine-tuning 
1. Data preparation
- Take a look at the samples at data/samples
2.Edit config 
- Edit batch_size, hyper params at manga_ocr_dev/training/train.py
- Edit  (MAIN_ROOT,...) at manga_ocr_dev/env.py 
    - DATA_SYNTHETIC_ROOT: Update MAIN_ROOT path, DATA_SYNTHETIC_ROOT will update  automatelly
    - TRAIN_ROOT: Save folder
    - Important: DATA_SYNTHETIC_ROOT_VER is the post prefix name 
    - (for ex: if DATA_SYNTHETIC_ROOT_VER='2' the actual train path is train2 and val path is val2 )
    - Recommend take a look at manga_ocr_dev/training/dataset.py  to see how the dataset read  from disk
    - Recommend take a look at manga_ocr_dev/env.py  to see how the path transform


```
- MAIN_ROOT
    - synthetic ( auto join the path in file  manga_ocr_dev/env.py)
            - train2 (if DATA_SYNTHETIC_ROOT_VER = '2')
                --images
                --label.csv
            - val2 
                --images
                --label.csv
```
- Label.csv file
    - source,vertical and font_path is any value we don't use this now ( i will update the code)
    - id : Name of image (if the img name is 111.png the actual value is 111 - not contains the value after '.' )
    - Text is text of Image
    - Note all image have to end with ".png" Which i hard code in manga_ocr_dev/training/dataset.py
3. Run trainning command

- Single GPU
```bash
manga_ocr_dev/training/train.py
```

- Multi GPUS
```bash
CUDA_VISIBLE_DEVICES=2,3 manga_ocr_dev/training/train.py
```
# Evaluate 
```bash
python  manga_ocr_dev/training/eval_model.py --device 0
```
# References
May contains some version errors that i updated in my code
- [kha-white/manga-ocr](https://github.com/kha-white/manga-ocr): A powerful OCR system for extracting text from manga images using deep learning.
