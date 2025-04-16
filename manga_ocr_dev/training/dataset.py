import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from manga_ocr_dev.env import  DATA_SYNTHETIC_ROOT
import os

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

		print(f"Initializing dataset {split}...")

	
		root_folder = DATA_SYNTHETIC_ROOT / split
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
		pixel_values = processor(img, return_tensors="pt").pixel_values
		return pixel_values.squeeze()

	@staticmethod
	def get_transforms():
		t_medium = A.Compose(
			[
				A.Rotate(5, border_mode=cv2.BORDER_REPLICATE, p=0.2),
				A.Perspective((0.01, 0.05), pad_mode=cv2.BORDER_REPLICATE, p=0.2),
				A.Blur(p=0.2),
				A.RandomBrightnessContrast(p=0.5),
				A.ToGray(p= 1),
			]
		)

		t_heavy = A.Compose(
			[
				A.Rotate(10, border_mode=cv2.BORDER_REPLICATE, p=0.2),
				A.Blur((4, 9), p=0.5),
				A.RandomBrightnessContrast(0.8, 0.8, p=1),
				A.ToGray(p = 1),
			]
		)

		return t_medium, t_heavy


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
