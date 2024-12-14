'''from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import onnx

# Load pre-trained BART model and tokenizer
model_name = 'facebook/bart-large-cnn'  
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
#tokenizer.save_pretrained('./tokenizer')

input_text = "Your long text here that needs to be summarized."
inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)

# Convert model to ONNX
onnx_path = "bart_model.onnx"
torch.onnx.export(model, 
                  (inputs["input_ids"],), 
                  onnx_path, 
                  input_names=["input_ids"], 
                  output_names=["summary_ids"], 
                  dynamic_axes={
                      "input_ids": {0: "batch_size", 1: "sequence"}, 
                      "summary_ids": {0: "batch_size", 1: "sequence"}
                  },
                  opset_version=14  
)

print("Model has been successfully exported to ONNX!")'''

from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import json

# Load pre-trained BART model 
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

#  tokenizer's vocab in a simple JSON file
token_json_path = 'token.json'

# Save the vocabulary dictionary 
with open(token_json_path, 'w') as f:
    json.dump(tokenizer.get_vocab(), f)

print(f"Tokenizer vocab saved to {token_json_path}")

# Input text for summarization
input_text = "Your long text here that needs to be summarized."
inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)

# Convert model to ONNX
onnx_path = "bart_model.onnx"
torch.onnx.export(model, 
                  (inputs["input_ids"],), 
                  onnx_path, 
                  input_names=["input_ids"], 
                  output_names=["summary_ids"], 
                  dynamic_axes={
                      "input_ids": {0: "batch_size", 1: "sequence"}, 
                      "summary_ids": {0: "batch_size", 1: "sequence"}
                  },
                  opset_version=14)

print("Model has been successfully exported to ONNX!")
