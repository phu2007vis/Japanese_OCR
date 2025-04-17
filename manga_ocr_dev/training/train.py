import fire
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
import os 
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from manga_ocr_dev.env import *


from manga_ocr_dev.training.dataset import MangaDataset
from manga_ocr_dev.training.get_model import get_model
from manga_ocr_dev.training.metrics import Metrics
from manga_ocr_dev.training.utils import visualize

def run():
    MAX_SEQUENCE_LENGTH = 16
    model, processor = get_model(max_length=MAX_SEQUENCE_LENGTH)

    train_dataset = MangaDataset(processor, "train", MAX_SEQUENCE_LENGTH, augment=True,)
    eval_dataset = MangaDataset(processor, "val", MAX_SEQUENCE_LENGTH, augment=False)
    try:
        visualize(MangaDataset(processor, "train", MAX_SEQUENCE_LENGTH, augment=True),phase = 'train')
        visualize(MangaDataset(processor, "val", MAX_SEQUENCE_LENGTH, augment=False),phase = 'val')
    except:
        pass
    # import pdb;pdb.set_trace()
    
    metrics = Metrics(processor)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        eval_strategy="steps",
        save_strategy="steps",
        per_device_train_batch_size=200,
        per_device_eval_batch_size=2000//2,
        fp16=True,
        fp16_full_eval=True,
        dataloader_num_workers=2,
        output_dir=TRAIN_ROOT,
        logging_steps=100,
        report_to="none",
        save_steps=100,
        eval_steps=100,
        num_train_epochs=500,
        run_name=RUN_NAME,
        eval_on_start=False,
        
        # label_smoothing_factor = 0.2
        #resume_from_checkpoint= TRAIN_ROOT / 'checkpoint-30000'
    )
 
    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=processor.image_processor,
        args=training_args,
        compute_metrics=metrics.compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        
    )
    # print("🔍 Evaluation before training:")
    # initial_metrics = trainer.evaluate()
    # print(initial_metrics)
    # exit()
    # print("🔍 Evaluation before training:")
    # initial_metrics = trainer.evaluate()
    # print(initial_metrics)
    # exit()
    
    trainer.train(
        # resume_from_checkpoint= True
    )


if __name__ == "__main__":
    fire.Fire(run)