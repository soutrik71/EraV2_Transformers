from torch.utils.data import Dataset, DataLoader, random_split
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase
import os
from S16_code.config import get_config
from S16_code.dataset import BilingualDataset


tokenizer_path = get_config()["tokenizer_path"]


# Iterating through dataset to extract the original sentence and its translation
def get_all_sentences(ds, lang):
    """Iterate over all sentences in the dataset and yield them."""
    for pair in ds:
        yield pair["translation"][lang]


def build_tokenizer(ds, lang):
    """Function to build a tokenizer for the given language and dataset"""
    if os.path.exists(os.path.join(tokenizer_path, f"tokenizer_{lang}.json")):
        print(f"Tokenizer for {lang} already exists")
        tokenizer = Tokenizer.from_file(
            os.path.join(tokenizer_path, f"tokenizer_{lang}.json")
        )
        return tokenizer

    print(f"Building tokenizer for {lang}")
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.normalizer = Lowercase()
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
    )
    tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
    os.makedirs("./tokenizer", exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_path, f"tokenizer_{lang}.json"))
    return tokenizer


def get_ds(ds_raw, config):
    """Function to create the train and validation dataset for the transformer model"""
    # Build tokenizers
    tokenizer_src = build_tokenizer(ds_raw, config["src_lang"])
    tokenizer_tgt = build_tokenizer(ds_raw, config["tgt_lang"])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(config["split_ratio"] * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["src_lang"],
        config["tgt_lang"],
        config["seq_len"],
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["src_lang"],
        config["tgt_lang"],
        config["seq_len"],
    )

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["src_lang"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["tgt_lang"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
