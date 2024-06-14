import gradio as gr
import torch
import os

#import model and the configuration
from model_gpt import GPT, GPTConfig

#set the device 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#load the model

checkpoint = torch.load('ckpt.pt', map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

#load the dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)

# gradio function 
def generate_output(length):
  context = torch.zeros((1, 1), dtype=torch.long, device=device)
  output_sequence = decode(model.generate(context, max_new_tokens=length)[0].tolist())
  return output_sequence

# instance gradio applications
title = "Shakespeare Text Generation"
description = "Model that generates text in the style of William Shakespeare."

demo = gr.Interface(
    fn = generate_output,
    inputs = [gr.Number(value = 50,label = "Sequence Length",info = "Length of the sample sequence you wish to generate.")],
    outputs = [gr.TextArea(lines = 5,label="Sequence Output")],
    title = title,
    description = description
)

# launch interface
demo.launch()