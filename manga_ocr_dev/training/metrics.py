import numpy as np
from  evaluate import load 


class Metrics:
    def __init__(self, processor):
        self.cer_metric = load("cer")
        self.processor = processor

    def compute_metrics(self, pred):
        label_ids = pred.label_ids
        pred_ids = pred.predictions
       
        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = np.array(["".join(text.split()) for text in pred_str])
        label_str = np.array(["".join(text.split()) for text in label_str])
        # for pred,label in zip(pred_str,label_str):
        #     if pred != label:
        #         import pdb;pdb.set_trace()
        results = {}
        
        try:
            results["cer"] = self.cer_metric.compute(predictions=pred_str, references=label_str)
        except Exception as e:
            print(e)
            print(pred_str)
            print(label_str)
            results["cer"] = 0
        results["accuracy"] = (pred_str == label_str).mean()

        return results
