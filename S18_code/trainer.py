from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn as nn
import torch
from tqdm import tqdm
from S18_code.dataset import causal_mask


# The idea is learning rate should be high initially and then decrease as the training progresses and more the embeddings are learned lower the learning rate should be
def custom_lr(step, dim_embed, warmup_steps):
    """Learning rate schedule as per the paper"""
    lr = (dim_embed**-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
    return lr


class CustomLRScheduler(_LRScheduler):
    """Custom learning rate scheduler"""

    def __init__(
        self,
        optimizer: Optimizer,
        dim_embed: int,
        warmup_steps: int,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:

        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> float:
        lr = custom_lr(self._step_count, self.dim_embed, self.warmup_steps)
        return [lr] * self.num_param_groups


class TranslationLoss(nn.Module):
    """Custom loss function for translation task"""

    def __init__(self, pad_idx, label_smoothing, tokenizer_tgt):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx, label_smoothing=label_smoothing
        )
        self.vocab_size = tokenizer_tgt.get_vocab_size()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        logits = logits.view(-1, self.vocab_size)
        labels = labels.view(-1).long()
        return self.criterion(logits, labels)


def run_training_loop_basic(
    model, train_dataloader, optimizer, loss_fn, device, global_step, scheduler=None
):
    """Training loop for the model"""

    model.train()
    train_batch_writer = tqdm(
        train_dataloader, total=len(train_dataloader), desc="Training"
    )
    total_loss = 0.0
    num_batches = len(train_dataloader)

    for batch in train_batch_writer:
        encoder_input = batch["encoder_input"].to(device, non_blocking=True)
        decoder_input = batch["decoder_input"].to(device, non_blocking=True)
        encoder_mask = batch["encoder_mask"].to(device, non_blocking=True)
        decoder_mask = batch["decoder_mask"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        # Forward pass
        optimizer.zero_grad(set_to_none=True)  # Clear gradients before forward pass
        encoder_output = model.encode(encoder_input, encoder_mask)
        decoder_output = model.decode(
            encoder_output, encoder_mask, decoder_input, decoder_mask
        )
        projection_output = model.project(decoder_output)

        # Computing loss between model's output and true labels
        loss = loss_fn(projection_output, label)
        total_loss += loss.item()

        # Updating progress bar
        train_batch_writer.set_postfix({"loss": f"{loss.item():6.3f}"})

        loss.backward()

        # Updating parameters based on the gradients
        optimizer.step()

        global_step += 1

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

    # Average training loss
    epoch_loss = total_loss / num_batches
    train_batch_writer.set_postfix({"Epoch Training loss": f"{epoch_loss:6.3f}"})
    return epoch_loss, global_step


def run_training_loop_opt(
    model,
    train_dataloader,
    optimizer,
    loss_fn,
    device,
    global_step,
    scaler,
    scheduler=None,
):
    """Training loop for the model with optimizations"""

    model.train()
    train_batch_writer = tqdm(
        train_dataloader, total=len(train_dataloader), desc="Training"
    )
    total_loss = 0.0
    num_batches = len(train_dataloader)

    for batch in train_batch_writer:
        encoder_input = batch["encoder_input"].to(device, non_blocking=True)
        decoder_input = batch["decoder_input"].to(device, non_blocking=True)
        encoder_mask = batch["encoder_mask"].to(device, non_blocking=True)
        decoder_mask = batch["decoder_mask"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # Clear gradients before forward pass

        # Forward pass within autocast context to enable mixed precision training
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )
            projection_output = model.project(decoder_output)
            loss = loss_fn(projection_output, label)

        # Scale the loss value to prevent underflow/overflow in mixed precision training
        scaler.scale(loss).backward()

        # Update parameters based on the gradients
        scaler.step(optimizer)

        # Update the scale for next iteration
        scaler.update()

        total_loss += loss.item()

        # Update progress bar with current batch loss
        train_batch_writer.set_postfix({"loss": f"{loss.item():.3f}"})

        global_step += 1

        # Adjust learning rate if using scheduler
        if scheduler is not None:
            scheduler.step()

    # Compute average training loss for the epoch
    epoch_loss = total_loss / num_batches
    train_batch_writer.set_postfix({"Epoch Training loss": f"{epoch_loss:.3f}"})
    return epoch_loss, global_step


def run_validation_loop(model, val_dataloader, loss_fn, device):
    """Validation loop for the model"""
    # Setting model to evaluation mode
    model.eval()

    # Progress bar for validation loop
    val_batch_writer = tqdm(
        val_dataloader, total=len(val_dataloader), desc="Validation"
    )
    total_loss = 0.0
    num_batches = len(val_dataloader)

    with torch.no_grad():
        for batch in val_batch_writer:
            # all the tensors are moved to the device
            encoder_input = batch["encoder_input"].to(device, non_blocking=True)
            decoder_input = batch["decoder_input"].to(device, non_blocking=True)
            encoder_mask = batch["encoder_mask"].to(device, non_blocking=True)
            decoder_mask = batch["decoder_mask"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            # Forward pass without calculating gradients
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )
            projection_output = model.project(decoder_output)

            # Computing loss between model's output and true labels
            loss = loss_fn(projection_output, label)
            total_loss += loss.item()

            val_batch_writer.set_postfix({"loss": f"{loss.item():6.3f}"})

    # average validation loss
    epoch_loss = total_loss / num_batches
    val_batch_writer.set_postfix({"Epoch Validation loss": f"{epoch_loss:6.3f}"})
    return epoch_loss


def greedy_decode(model, enocode, encode_mask, max_len, device, sos_idx, eos_idx):
    """Greedy decoding for the model"""
    # encoder_output
    encode_op = model.encode(enocode, encode_mask)

    # sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(enocode).to(device)

    while decoder_input.size(1) < max_len:
        # decoder mask
        decode_mask = causal_mask(decoder_input.size(1)).type_as(encode_mask).to(device)

        # decoder output
        decode_op = model.decode(encode_op, encode_mask, decoder_input, decode_mask)

        # project the output
        projection_op = model.project(decode_op[:, -1])

        # get next token
        _, next_word = torch.max(projection_op, dim=1)

        # append the next word to the decoder input and repeat the process
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(enocode).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )

        # break if eos token is predicted or max_len is reached
        if next_word.item() == eos_idx:
            break

    return decoder_input[:, 1:].squeeze(0)  # remove sos token from beginning


def run_inference_loop(
    model,
    val_dataloader,
    tokenizer_tgt,
    device,
    num_examples,
    metric,
    sos_idx,
    eos_idx,
    cushion=50,
):
    """Inference loop for the model"""
    # Set the model to evaluation mode
    model.eval()

    # Initialize counters and lists to store results
    count = 0
    source_texts, expected, predicted = [], [], []

    # Initialize tqdm for progress tracking
    val_batch_writer = tqdm(val_dataloader, total=len(val_dataloader), desc="Inference")

    # Disable gradient calculation during inference
    with torch.no_grad():
        for batch in val_batch_writer:
            # Stop iterating if desired number of examples is reached
            if count == num_examples:
                break

            # Move data to the appropriate device
            encoder_input = batch["encoder_input"].to(device, non_blocking=True)
            encoder_mask = batch["encoder_mask"].to(device, non_blocking=True)

            # Calculate maximum length for decoding
            max_len = encoder_input.shape[-1] + cushion

            # Ensure batch size is 1 for validation
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation."

            # Perform greedy decoding to generate the output sequence
            model_out = greedy_decode(
                model, encoder_input, encoder_mask, max_len, device, sos_idx, eos_idx
            )

            # Retrieve source and target texts from the batch
            source_text = batch["src_texts"][0]
            target_text = batch["tgt_texts"][0]

            # Decode the model output to text format
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # Print the source, target, and predicted texts
            val_batch_writer.write(f"SOURCE: {source_text}")
            val_batch_writer.write(f"TARGET: {target_text}")
            val_batch_writer.write(f"PREDICTED: {model_out_text}")

            # Append source, target, and predicted texts to respective lists
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Update counter for processed examples
            count += 1

        # Calculate BLEU score for the entire inference process
        acc = metric(expected, predicted)
        print(f"BLEU Score: {acc}")
        metric.reset()  # Reset the metric for future calculations
