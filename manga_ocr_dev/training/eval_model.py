import os
import sys
import argparse
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator

# add your project’s root to PYTHONPATH
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)

from manga_ocr_dev.env import *
from manga_ocr_dev.training.dataset import MangaDataset
from manga_ocr_dev.training.get_model import get_model
from manga_ocr_dev.training.metrics import Metrics

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation on the Manga OCR val split"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Which GPU to use (sets CUDA_VISIBLE_DEVICES)",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) Select GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # 2) Load model & processor
    MAX_SEQUENCE_LENGTH = 16
    model, processor = get_model(max_length=MAX_SEQUENCE_LENGTH)

    # 3) Prepare the eval dataset
    eval_dataset = MangaDataset(
        processor,
        split="val",
        max_target_length=MAX_SEQUENCE_LENGTH,
        augment=False,
    )
    metrics = Metrics(processor)

    # 4) Trainer arguments tailored for evaluation
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        per_device_eval_batch_size=50,
        fp16=True,
        fp16_full_eval=True,
        dataloader_num_workers=2,
        output_dir=TRAIN_ROOT,
        report_to="none",
    )

    # 5) Instantiate the Trainer for eval only
    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=processor.image_processor,
        args=training_args,
        compute_metrics=metrics.compute_metrics,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    # 6) Run evaluation
    print(f"🔍 Running evaluation on GPU {args.device} …")
    eval_metrics = trainer.evaluate()
    for key, value in eval_metrics.items():
        print(f"{key:30s} {value}")
    return eval_metrics

if __name__ == "__main__":
    main()
