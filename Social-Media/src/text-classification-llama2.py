import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import json
import pandas as pd
import numpy as np
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
from transformers import GenerationConfig
from transformers.generation.utils import GreedySearchDecoderOnlyOutput
from transformers import GenerationConfig
import textwrap
from tqdm.auto import tqdm

# change the directory to your llama 2 model
BASE_MODEL = r"/shared/4/models/llama2/pytorch-versions/llama-2-13b-chat"

model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"


PROMPT_TEMPLATE = f"""Question: Is the following comment talking about environmental issues of Lake Erie, such as water pollution? Answer 'True' or 'False' without explaining your reasoning.
Comment: {"[INPUT]"}
Answer:"""


def generate_batch_responses(text_inputs, model, tokenizer, template, batch_size=8):
    responses = []    
    device = model.device  # Use the model's device
    
    # Prepare batches of prompts
    for i in tqdm(range(0, len(text_inputs), batch_size), desc="Processing"):
        batch_prompts = text_inputs[i:i + batch_size]
        prompts = [template.replace("[INPUT]", text).replace("'", '') for text in batch_prompts]
        
        # Tokenize all prompts in the current batch and move to the correct device
        encoding = tokenizer(prompts, padding=True, return_tensors="pt", max_length=512, truncation=True)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        with torch.no_grad():
            batch_responses = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                temperature=0,
                # top_p=0.75,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                # do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # Decode each response in the batch
        for response in batch_responses.sequences:
            decoded_output = tokenizer.decode(response, skip_special_tokens=True)
            split_response = decoded_output.split("Answer:")[1].strip() if "Answer:" in decoded_output else decoded_output
            responses.append(split_response)

        # Optionally, save interim results
        if (i // batch_size) % 100 == 0:
            filename = f"../result/llama2-{i}.json"
            with open(filename, "w") as json_file:
                json.dump(responses, json_file)

    return responses


    
# read the target api dataset
with open('../data/all_RC.json', 'r') as json_file:
    json_objects = json.load(json_file)
    
text_inputs_raw = [obj['body'] for obj in json_objects]
text_inputs_raw = text_inputs_raw[20000:]

text_inputs = []
for text in text_inputs_raw:
    words = text.split()

    if len(words) > 450:
        words = words[-450:]

    truncated_text = " ".join(words)
    text_inputs.append(truncated_text)

responses = generate_batch_responses(
    text_inputs=text_inputs,
    model=model,
    tokenizer=tokenizer,
    template = PROMPT_TEMPLATE,
    batch_size=32  # Adjust batch size as needed based on available memory
)

 # Save the generated responses to a file
filename = "../result/llama2.json"
with open(filename, "w") as json_file:
    json.dump(responses, json_file)
