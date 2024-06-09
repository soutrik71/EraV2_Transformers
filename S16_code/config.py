import torch
from pathlib import Path


def get_config():
    return {
        "model_folder": "models",
        "batch_size": 16,
        "num_epochs": 10,
        "learning_rate": 10e-4,
        "seq_len": 350,
        "d_model": 512,
        "N": 6,
        "h": 8,
        "dropout_rate": 0.1,
        "d_ff": 2048,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "src_lang": "en",
        "tgt_lang": "it",
        "tokenizer_path": "./tokenizer",
        "datasource": "opus_books",
        "model_basename": "transformer_model",
        "split_ratio": 0.9,
    }


def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}_{epoch}.pt"
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    Path(f"{model_folder}").mkdir(parents=True, exist_ok=True)
    return str(Path(".") / model_folder / model_filename)
