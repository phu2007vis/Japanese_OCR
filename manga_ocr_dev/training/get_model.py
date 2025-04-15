from transformers import (
	AutoTokenizer,
	AutoImageProcessor,
	VisionEncoderDecoderModel,
	TrOCRProcessor,
	GenerationMixin,
)

import os

class PhuocModel(VisionEncoderDecoderModel, GenerationMixin):
    pass

class TrOCRProcessorCustom(TrOCRProcessor):
	"""The only point of this class is to bypass type checks of base class."""

	def __init__(self, feature_extractor, tokenizer):
		super().__init__(feature_extractor=feature_extractor, tokenizer=tokenizer)
		self.current_processor = self.feature_extractor


def get_model( pretrained_model_name_or_path =None ):

	if pretrained_model_name_or_path is None:
		file_location = os.path.abspath(__file__)
		pretrained_model_name_or_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(file_location))),'weights_main','phuoc')
	
	model = PhuocModel.from_pretrained(pretrained_model_name_or_path)

	image_processor = AutoImageProcessor.from_pretrained(pretrained_model_name_or_path, use_fast=True)
	tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
	processor = TrOCRProcessorCustom(image_processor, tokenizer)

	return model, processor
