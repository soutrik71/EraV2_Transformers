# Import necessary libraries
from S18_code.config import get_config
import os
from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase
import torch
from torch.utils.data import DataLoader, random_split
from S18_code.dataset import BilingualDataset, causal_mask
import functools

tokenizer_path = get_config()["tokenizer_path"]


def get_all_sentences(ds, lang):
    """Iterate over all sentences in the dataset and yield them."""
    for pair in ds:
        yield pair["translation"][lang]


def build_tokenizer(ds, lang):
    """Function to build a tokenizer for the given language and dataset"""
    print(f"Building tokenizer for {lang}")

    if os.path.exists(os.path.join(tokenizer_path, f"tokenizer_{lang}.json")):
        print(f"Tokenizer for {lang} already exists")
        tokenizer = Tokenizer.from_file(
            os.path.join(tokenizer_path, f"tokenizer_{lang}.json")
        )
        return tokenizer

    # Initialize a WordLevel tokenizer with an unknown token
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))

    # Use whitespace for tokenization
    tokenizer.pre_tokenizer = Whitespace()

    # Convert all text to lowercase to ensure consistency
    tokenizer.normalizer = normalizers.Sequence([Lowercase()])

    # Define special tokens and set minimum frequency for words
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
    )

    # Train tokenizer on the provided dataset
    tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)

    os.makedirs("./tokenizer", exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_path, f"tokenizer_{lang}.json"))

    return tokenizer


def collate_fn(batch, tokenizer_tgt):
    """Collate function to pad sequences and create masks for batching"""
    encoder_input = [sample["encoder_input"] for sample in batch]
    decoder_input = [sample["decoder_input"] for sample in batch]
    label = [sample["label"] for sample in batch]
    src_texts = [sample["src_text"] for sample in batch]
    tgt_texts = [sample["tgt_text"] for sample in batch]

    # Pad sequences for batching

    # encoder input should be of shape (batch_size, seq_len)
    encoder_input = torch.nn.utils.rnn.pad_sequence(
        encoder_input,
        batch_first=True,
        padding_value=tokenizer_tgt.token_to_id("[PAD]"),
    )

    # decoder input and label should be of shape (batch_size, seq_len)
    decoder_input = torch.nn.utils.rnn.pad_sequence(
        decoder_input,
        batch_first=True,
        padding_value=tokenizer_tgt.token_to_id("[PAD]"),
    )
    label = torch.nn.utils.rnn.pad_sequence(
        label, batch_first=True, padding_value=tokenizer_tgt.token_to_id("[PAD]")
    )

    # print(encoder_input.size())
    # print(decoder_input.size())
    # print(label.size())

    # Create masks - Very Tricky
    # Encoder mask should be of shape (batch_size, 1, 1, seq_len)
    encoder_mask = (
        (encoder_input != tokenizer_tgt.token_to_id("[PAD]"))
        .unsqueeze(-2)
        .unsqueeze(-2)
        .type(torch.int)
    )
    # print(encoder_mask.size())

    # Decoder mask should be of shape (batch_size, 1, seq_len, seq_len)
    decoder_mask = (
        (decoder_input != tokenizer_tgt.token_to_id("[PAD]"))
        .unsqueeze(1)
        .type(torch.int)
        & (causal_mask(decoder_input.size(1)).unsqueeze(0))
    ).unsqueeze(1)

    # print(decoder_mask.size())

    return {
        "encoder_input": encoder_input,  # (batch_size, seq_len)
        "decoder_input": decoder_input,  # (batch_size, seq_len)
        "label": label,  # (batch_size, seq_len)
        "encoder_mask": encoder_mask,  # (batch_size, 1, 1,seq_len)
        "decoder_mask": decoder_mask,  # (batch_size,1, seq_len, seq_len)
        "src_texts": src_texts,  # List of source texts
        "tgt_texts": tgt_texts,  # List of target texts
    }


def get_ds(ds_raw, config):
    """Function to create the train and validation dataset for the transformer model"""
    # Build tokenizers
    tokenizer_src = build_tokenizer(ds_raw, config["src_lang"])
    tokenizer_tgt = build_tokenizer(ds_raw, config["tgt_lang"])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(config["split_ratio"] * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    print("Creating the tokenized-dataset")
    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["src_lang"],
        config["tgt_lang"],
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["src_lang"],
        config["tgt_lang"],
    )

    print(
        f"Dataloaders created with {len(train_ds)} training and {len(val_ds)} validation samples"
    )
    train_dataloader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        collate_fn=functools.partial(collate_fn, tokenizer_tgt=tokenizer_tgt),
        shuffle=config["shuffle"],
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=1,
        collate_fn=functools.partial(collate_fn, tokenizer_tgt=tokenizer_tgt),
    )

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt
