from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer
)

import torch


def load_model():

    model_name = "gpt2"

    tokenizer = GPT2Tokenizer.from_pretrained(
        model_name
    )

    model = GPT2LMHeadModel.from_pretrained(
        model_name,

        # CRITICAL FIX
        attn_implementation="eager"
    )

    device = torch.device("cpu")

    model.to(device)

    return model, tokenizer, device