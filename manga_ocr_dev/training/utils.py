import numpy as np
import torch
from torchinfo import summary


def encoder_summary(model, batch_size=4):
    img_size = model.config.encoder.image_size
    return summary(
        model.encoder,
        input_size=(batch_size, 3, img_size, img_size),
        depth=3,
        col_names=["output_size", "num_params", "mult_adds"],
        device="cpu",
    )


def decoder_summary(model, batch_size=4):
    img_size = model.config.encoder.image_size
    encoder_hidden_shape = (
        batch_size,
        (img_size // 16) ** 2 + 1,
        model.config.decoder.hidden_size,
    )
    decoder_inputs = {
        "input_ids": torch.zeros(batch_size, 1, dtype=torch.int64),
        "attention_mask": torch.ones(batch_size, 1, dtype=torch.int64),
        "encoder_hidden_states": torch.rand(encoder_hidden_shape, dtype=torch.float32),
        "return_dict": False,
    }
    return summary(
        model.decoder,
        input_data=decoder_inputs,
        depth=4,
        col_names=["output_size", "num_params", "mult_adds"],
        device="cpu",
    )


def tensor_to_image(img):
    return ((img.cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)

def visualize(dataset,phase,save_folder = None):
    import cv2
    if save_folder == None:
        import os 

        save_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        save_folder = os.path.join(save_folder,'visualize',phase)

    os.makedirs(save_folder,exist_ok=True)
    dataset.return_text = True
    for i in range(20):
        encoding = dataset[i]
        img = tensor_to_image(encoding['pixel_values'])
        text = encoding['text']
        img_name = f"{text}_{i}.png"
        img_path  = os.path.join(save_folder,img_name)
        cv2.imwrite(img_path,img)
    dataset.return_text = False 
    print(f"Save visualize at {save_folder}")