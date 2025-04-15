import re
from pathlib import Path
import os
import jaconv
import torch
from PIL import Image
from loguru import logger
from transformers import ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel, GenerationMixin,ViTForImageClassification,ViTConfig

from transformers import (
    AutoTokenizer,
    VisionEncoderDecoderModel,
)

class PhuocModel(VisionEncoderDecoderModel, GenerationMixin):
    pass


class MangaOcr:
    def __init__(self, pretrained_model_name_or_path=None, force_cpu=False):
        logger.info(f"Loading OCR model from {pretrained_model_name_or_path}")
        if pretrained_model_name_or_path is None:
            
            file_location = os.path.abspath(__file__)
            pretrained_model_name_or_path = os.path.join(os.path.dirname(os.path.dirname(file_location)),'weights_main','phuoc')
            
        self.processor = ViTImageProcessor.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = PhuocModel.from_pretrained(pretrained_model_name_or_path)

        if not force_cpu and torch.cuda.is_available():
            logger.info("Using CUDA")
            self.model.cuda()
        elif not force_cpu and torch.backends.mps.is_available():
            logger.info("Using MPS")
            self.model.to("mps")
        else:
            logger.info("Using CPU")

        logger.info("OCR ready")

    def __call__(self, img_or_path,verbose = True):
        
        if isinstance(img_or_path, str) or isinstance(img_or_path, Path):
            img = Image.open(img_or_path)
        elif isinstance(img_or_path, Image.Image):
            img = img_or_path
        else:
            raise ValueError(f"img_or_path must be a path or PIL.Image, instead got: {img_or_path}")

        img = img.convert("L").convert("RGB")

        x = self._preprocess(img)
        x = self.model.generate(x[None].to(self.model.device), max_length=300)[0].cpu()
        x = self.tokenizer.decode(x, skip_special_tokens=True)

        x = post_process(x)

        if verbose:
            print(x)

        return x

    def _preprocess(self, img):
        pixel_values = self.processor(img, return_tensors="pt").pixel_values
        return pixel_values.squeeze()


def post_process(text):
    text = "".join(text.split())
    text = text.replace("…", "...")
    text = re.sub("[・.]{2,}", lambda x: (x.end() - x.start()) * ".", text)
    text = jaconv.h2z(text, ascii=True, digit=True)

    return text
