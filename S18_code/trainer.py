import torch
from torch import nn
from S16_code.dataset import causal_mask
from tqdm import tqdm
import os


def greedy_decode(
    model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device
):
    """Function to generate the target sequence using greedy decoding
    It happens in the following way:
    1. The encoder encodes the source sequence
    2. The decoder generates the target sequence token by token
    3. The token with the highest probability is selected as the next token
    4. The loop continues until the End of Sentence token is generated or the maximum length is reached
    5. The target sequence is returned
    """
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation_loop(
    model, val_dataloader, tokenizer_src, tokenizer_tgt, device, seq_len, num_examples
):
    """Function to run the validation of the model
    It generates the target sequence for a given source sequence and prints the source, target, and predicted sequences

    """
    model.eval()
    count = (
        0  # Initializing counter to keep track of how many examples have been processed
    )
    source_texts = []
    expected = []
    predicted = []

    val_batch_writer = tqdm(
        val_dataloader, total=len(val_dataloader), desc="Validation"
    )
    with torch.no_grad():
        for batch in val_batch_writer:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            # Ensuring that the batch_size of the validation set is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation."

            model_out = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                seq_len,
                device,
            )

            # Retrieving source and target texts from the batch
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]  # True translation

            # Decoding the model output
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # Printing results
            val_batch_writer.write(f"SOURCE: {source_text}")
            val_batch_writer.write(f"TARGET: {target_text}")
            val_batch_writer.write(f"PREDICTED: {model_out_text}")

            # Appending the source, target, and predicted sequences to the respective lists
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            if count == num_examples:
                break


def run_training_loop(
    model, train_dataloader, tokenizer_tgt, optimizer, loss_fn, device, global_step
):

    model.train()
    train_batch_writer = tqdm(
        train_dataloader, total=len(train_dataloader), desc="Training"
    )

    for batch in train_batch_writer:
        encoder_input = batch["encoder_input"].to(device)
        decoder_input = batch["decoder_input"].to(device)
        encoder_mask = batch["encoder_mask"].to(device)
        decoder_mask = batch["decoder_mask"].to(device)
        label = batch["label"].to(device)

        encoder_output = model.encode(encoder_input, encoder_mask)
        decoder_output = model.decode(
            encoder_output, encoder_mask, decoder_input, decoder_mask
        )
        projection_output = model.project(decoder_output)

        # Computing loss between model's output and true labels
        loss = loss_fn(
            projection_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
        )

        # Updating progress bar
        train_batch_writer.set_postfix({"loss": f"{loss.item():6.3f}"})

        loss.backward()

        # Updating parameters based on the gradients
        optimizer.step()

        # Clearing the gradients to prepare for the next batch
        optimizer.zero_grad(set_to_none=True)

        global_step += 1

    epoch_loss = loss.cpu().detach().numpy()
    return epoch_loss, global_step


def save_model(epoch, model, optimizer, global_step, model_filename):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
        },
        model_filename,
    )
