from transformers import (
	AutoTokenizer,
	AutoImageProcessor,
	TrOCRProcessor,
	
)
from copy import deepcopy
from timm.layers import resample_abs_pos_embed
import os
from model import PhuocModel,PhuocViTEmbeddings

class TrOCRProcessorCustom(TrOCRProcessor):
	"""The only point of this class is to bypass type checks of base class."""

	def __init__(self, feature_extractor, tokenizer):
		super().__init__(feature_extractor=feature_extractor, tokenizer=tokenizer)
		self.current_processor = self.feature_extractor
 
 

def replace_encoder(model, img_size,new_patch_size = (16,16), max_label_length=16):
	"""
	Replace the encoder's embedding layer in the given model with a new one
	adapted to a specific image size and max label length.

	Parameters:
		model (nn.Module): The original model with an encoder to replace.
		img_size (tuple): The target image size (height, width).
		max_label_length (int): Maximum label length for position query adjustment.

	Returns:
		nn.Module: The updated model with a new encoder embedding.
	"""
	# Update encoder configuration
	encoder_config = model.config.encoder
	encoder_config.image_size = img_size
	encoder_config.patch_size = new_patch_size
	
	

	# Create new embedding with updated config
	new_embedder = PhuocViTEmbeddings(encoder_config)

	# Copy original state_dict
	original_state_dict = model.encoder.embeddings.state_dict()
	original_weight = original_state_dict['position_embeddings']

	new_state_dict = {}

	for key, old_value in original_state_dict.items():
		if old_value.shape != new_embedder.state_dict()[key].shape:
			if key == 'pos_queries':
				# Adjust pos_queries length
				new_value = deepcopy(old_value[:, :max_label_length + 1, :])
			elif key == 'position_embeddings':
				# Interpolate position embeddings for new image size
				new_value = model.encoder.embeddings.interpolate_pos_encoding(
					original_weight, height=img_size[1], width=img_size[0]
				)
				print(f"Interpolated and copied: {key}")
			else:
				new_value = old_value
		else:
			new_value = old_value

		new_state_dict[key] = new_value

	# Load the updated state dict into the new embedding
	new_embedder.load_state_dict(new_state_dict)

	# Replace the model's embedding with the new one
	model.encoder.embeddings = new_embedder

	return model

def get_model( pretrained_model_name_or_path =None,max_length=None,img_size = (64,240) ):

	if pretrained_model_name_or_path is None:
		file_location = os.path.abspath(__file__)
		pretrained_model_name_or_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(file_location))),'weights_main','phuoc')
	
	model = PhuocModel.from_pretrained(pretrained_model_name_or_path)
	replace_encoder(model,img_size)
	image_processor = AutoImageProcessor.from_pretrained(pretrained_model_name_or_path, use_fast=True)
	new_h,new_w = model.config.encoder.image_size
	image_processor.size = {"height": new_h, "width": new_w}
	tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
	processor = TrOCRProcessorCustom(image_processor, tokenizer)
	
	if max_length is None:
		raise ValueError("max_length cannot be None")

	model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
	model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
	model.generation_config.max_length = max_length
	model.generation_config.early_stopping = True
	model.generation_config.num_beams = 4
	model.generation_config.no_repeat_ngram_size = 3
	model.generation_config.length_penalty = 2.0
	
 
	return model, processor
