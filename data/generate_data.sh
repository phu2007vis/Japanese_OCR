#!/bin/bash
#SBATCH --job-name=Image_captioning
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --account=ddt_acc23

#SBATCH --output=logs/%x_%j_%D.out
#SBATCH --error=logs/%x_%j_%D.err

squeue --me
cd /work/21013187/phuoc/Japanese_OCR
module load python 


python /work/21013187/phuoc/TextRecognitionDataGenerator4/trdg_phuoc/generators/from_strings.py \
 --output_dir=/work/21013187/phuoc/Japanese_OCR/data/synthetic/val3/images \
 -c 2 \
 --aug \
 --background_dir=/work/21013187/phuoc/TextRecognitionDataGenerator4/trdg_phuoc/images \
 --color_path=/work/21013187/phuoc/TextRecognitionDataGenerator4/trdg_phuoc/font_colors.txt \
 --remove_exsist \
 --font_dir=/work/21013187/phuoc/Japanese_OCR/data/fonts/ \
 --vocab_type='file' \
 --vocab='/work/21013187/phuoc/Japanese_OCR/assets/vocab.csv' 

# python /work/21013187/phuoc/TextRecognitionDataGenerator4/trdg_phuoc/generators/from_strings.py \
#  --output_dir=/work/21013187/phuoc/Japanese_OCR/data/synthetic/train3/images \
#  -c 10 \
#  --background_dir=/work/21013187/phuoc/TextRecognitionDataGenerator4/trdg_phuoc/images \
#  --color_path=/work/21013187/phuoc/TextRecognitionDataGenerator4/trdg_phuoc/font_colors.txt \
#  --remove_exsist \
#  --font_dir=/work/21013187/phuoc/Japanese_OCR/data/fonts/ \
#  --vocab_type='file' \
#  --vocab='/work/21013187/phuoc/Japanese_OCR/assets/vocab.csv' 