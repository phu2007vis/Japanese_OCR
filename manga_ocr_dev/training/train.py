import fire
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
import os 
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from manga_ocr_dev.env import TRAIN_ROOT
from manga_ocr_dev.training.dataset import MangaDataset
from manga_ocr_dev.training.get_model import get_model
from manga_ocr_dev.training.metrics import Metrics
from manga_ocr_dev.training.utils import visualize

def run(
    run_name="debug",
    max_length = 16,
    batch_size=200,
    num_epochs=500,
    logging_steps = 100,
    fp16=True,
    save_steps=200,
    eval_steps=200,
):

    model, processor = get_model(max_length=max_length)

    train_dataset = MangaDataset(processor, "train", max_length, augment=True,)
    eval_dataset = MangaDataset(processor, "val", max_length, augment=False)
    try:
        visualize(MangaDataset(processor, "train", max_length, augment=True),phase = 'train')
        visualize(MangaDataset(processor, "val", max_length, augment=False),phase = 'val')
    except:
        pass
    # print(len(train_dataset))
    # print(len(eval_dataset))
    
    # import pdb;pdb.set_trace()
    
    metrics = Metrics(processor)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        eval_strategy="steps",
        save_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size//2,
        fp16=fp16,
        fp16_full_eval=fp16,
        dataloader_num_workers=2,
        output_dir=TRAIN_ROOT,
        logging_steps=logging_steps,
        report_to="none",
        save_steps=save_steps,
        eval_steps=eval_steps,
        num_train_epochs=num_epochs,
        run_name=run_name,
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