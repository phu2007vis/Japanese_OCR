# TODO

1. Scrip for text generation
2. New pretrained weights

# Test 
![plot](data/visualize.png)
# Some sample synthetic data
![plot](data/595.png)
# Key requirements
transformers >= 4.5x 
python 3.11 is recommended which is my case (i also test run well with python 3.9)
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
    - Important: DATA_SYNTHETIC_ROOT_VER is the post prefix name (for ex: if DATA_SYNTHETIC_ROOT_VER='2' the actual train path is train2 and val path is val2 )
    - Recommend take a look at manga_ocr_dev/env.py  to see how the path transform


```

```

CUDA_VISIBLE_DEVICES
# References
May contains some version errors that i updated in my code
- [kha-white/manga-ocr](https://github.com/kha-white/manga-ocr): A powerful OCR system for extracting text from manga images using deep learning.
