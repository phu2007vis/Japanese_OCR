import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from manga_ocr_dev.env import  DATA_SYNTHETIC_ROOT,DATA_SYNTHETIC_ROOT_VER
import os
import random

global kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

class MangaDataset(Dataset):
	def __init__(
		self,
		processor,
		split,
		max_target_length,
		# phase,
		limit_size=None,
		augment=False,
	  	
	):
		self.processor = processor
		self.max_target_length = max_target_length

		data = []
		
		print(f"Initializing dataset {split +str(DATA_SYNTHETIC_ROOT_VER)}...")

	
		root_folder = DATA_SYNTHETIC_ROOT / (split+str(DATA_SYNTHETIC_ROOT_VER))
		csv_path = root_folder / 'label.csv'
		
		df = pd.read_csv(csv_path)
		df = df.dropna()
		df["path"] = df.id.apply(lambda x: str(root_folder/'images'/ f"{x}.png"))
		df = df[["path", "text"]]
		df["synthetic"] = True
  
		for path in df.path:
			assert(os.path.exists(path))
			data.append(df)


		data = data[0]
		if limit_size:
			data = data.iloc[:limit_size]
		self.data = data

		print(f"Dataset {split}: {len(self.data)}")
		self.return_text = False
		self.augment = augment
		self.transform_medium, self.transform_heavy = self.get_transforms()
		
  
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		
		sample = self.data.loc[idx]
		text = sample.text
		# print(sample.path)

		if self.augment:
			medium_p = 0.8
			heavy_p = 0.02
			transform_variant = np.random.choice(
				["none", "medium", "heavy"],
				p=[1 - medium_p - heavy_p, medium_p, heavy_p],
			)
			transform = {
				"none": None,
				"medium": self.transform_medium,
				"heavy": self.transform_heavy,
			}[transform_variant]
		else:
			transform = None

		pixel_values = self.read_image(self.processor, sample.path, transform)
		labels = self.processor.tokenizer(
			text,
			padding="max_length",
			max_length=self.max_target_length,
			truncation=True,
		).input_ids
		
		labels = np.array(labels)
		# important: make sure that PAD tokens are ignored by the loss function
		labels[labels == self.processor.tokenizer.pad_token_id] = -100
		
		encoding = {
			"pixel_values": pixel_values,
			"labels": torch.tensor(labels).type(torch.LongTensor) ,
		}
		if self.return_text:
			encoding['text'] = text
		return encoding

	@staticmethod
	def read_image(processor, path, transform=None):

		img = cv2.imread(str(path))
		if img is None:
			print(path)
		if transform is None:
			transform = A.ToGray(p = 1)
			img = transform(image=img)["image"]
		else: 
			img = transform(image=img)["image"]
			
			global kernel
			if random.randint(1, 5) == 1:
				# erosion because the image is not inverted
				img = cv2.dilate(img, kernel,iterations=random.randint(1, 1))
			
			elif random.randint(1, 5) == 1:
				img = cv2.erode(img, kernel, iterations=random.randint(1, 1))

		pixel_values = processor.image_processor(img, return_tensors="pt").pixel_values
		return pixel_values.squeeze()

	@staticmethod
	def get_transforms():
		t_medium =  A.Compose(
				[
		
				A.OneOf([
						A.ShiftScaleRotate(shift_limit=0, scale_limit=0.03, rotate_limit=4, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255),p=1),
						A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0, rotate_limit=4, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255),p=1),
						A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.03, rotate_limit=4, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255),p=1),        
				], p=0.5),
				
				A.AdditiveNoise (noise_type='gaussian', spatial_mode='per_pixel', p= 0.3,approximation = 0.65),
				A.Blur(blur_limit=5,p=0.33),
				A.OneOf([
							A.PixelDropout(drop_value=255,p=1),
							A.PixelDropout(p=1),
						], p=0.1),
				
				A.OneOf(
                    [
                        A.Downscale(0.25, 0.5, interpolation=cv2.INTER_LINEAR),
                        A.Downscale(0.25, 0.5, interpolation=cv2.INTER_NEAREST),
                    ],
                    p=0.1,
                ),
     			A.Sharpen(p=0.1),
			
					
				
			]
		)
	


		# t_heavy = A.Compose(
		# 	[
		# 		A.Rotate(limit=5, border_mode=cv2.BORDER_REPLICATE, p=0.2),
		# 		A.Perspective(scale=(0.005, 0.005), border_mode=cv2.BORDER_REPLICATE, p=0.2),
		# 		# A.InvertImg(p=0.05),
		# 		A.OneOf(
		# 			[
		# 				A.Downscale(scale_range=(0.25, 0.5),
		# 		  					interpolation_pair={'downscale': cv2.INTER_NEAREST, 'upscale': cv2.INTER_LINEAR}),
		# 			],
		# 			p=0.1,
		# 		),
		# 		A.Blur(blur_limit=(3, 9), p=0.5),
		# 		A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
		# 		A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=1),
		# 		A.GaussNoise(std_range=(0.124,  0.392), p=0.3),
		# 		A.ImageCompression(quality_range=(1, 10), p=0.5),
		# 		A.ToGray(p=1.0),
		# 	]
		# )

		return t_medium, t_medium

	

if __name__ == "__main__":
	from manga_ocr_dev.training.get_model import get_processor
	from manga_ocr_dev.training.utils import tensor_to_image

	encoder_name = "facebook/deit-tiny-patch16-224"
	decoder_name = "cl-tohoku/bert-base-japanese-char-v2"

	max_length = 300

	processor = get_processor(encoder_name, decoder_name)
	ds = MangaDataset(processor, "train", max_length, augment=True)

	for i in range(20):
		sample = ds[0]
		img = tensor_to_image(sample["pixel_values"])
		tokens = sample["labels"]
		tokens[tokens == -100] = processor.tokenizer.pad_token_id
		text = "".join(processor.decode(tokens, skip_special_tokens=True).split())

		print(f"{i}:\n{text}\n")
		plt.imshow(img)
		plt.show()
