import torch
from torch.utils.data import Dataset


def causal_mask(size):
    """Create a causal mask to ensure each position can attend to previous positions"""
    # The causal mask ensures that each token can only attend to previous tokens (for autoregressive decoding)
    mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.int)
    return mask == 0


class BilingualDataset(Dataset):
    """Custom Dataset for bilingual translation data"""

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang):
        super().__init__()

        # Initialize dataset and tokenizers
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Token IDs for special tokens
        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.ds)

    def __getitem__(self, idx):
        """Return a sample from the dataset at the given index"""
        src_target_pair = self.ds[idx]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        # Tokenize source and target texts
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add <SOS> and <EOS> tokens to the encoder input
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
            ],
            dim=0,
        )

        # Add <SOS> token to the decoder input
        decoder_input = torch.cat(
            [self.sos_token, torch.tensor(dec_input_tokens, dtype=torch.int64)], dim=0
        )

        # Add <EOS> token to the label
        label = torch.cat(
            [torch.tensor(dec_input_tokens, dtype=torch.int64), self.eos_token], dim=0
        )

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
