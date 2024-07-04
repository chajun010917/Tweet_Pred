import pandas as pd
from transformers import T5Tokenizer, T5Model
import re
import torch
from safetensors.torch import load_model
from custom_T5_model import CustomT5Model

#Data Load
file_path = './Q2_20230202_majority_top_30.csv' #Both load/save file path 
file = pd.read_csv(file_path)

#Data Cleaning
def clean_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)  
    tweet = re.sub(r'@\w+', '', tweet)  
    tweet = re.sub(r'#\w+', '', tweet)  
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    tweet = re.sub(r'[^\w\s,]', '', tweet)
    return tweet

file['clean_tweet'] = file['tweet'].apply(clean_tweet)

base_model = T5Model.from_pretrained("google/flan-t5-large")
tokenizer = T5Tokenizer.from_pretrained("./model_fe")

model = CustomT5Model(base_model)
load_model(model, './model_fe/classifier.safetensors')
model.eval()

def get_pred(tweet, tokenizer, model):
    inputs = tokenizer(tweet, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
    with torch.no_grad():
        logits = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        if isinstance(logits, tuple):
            logits = logits[1]
        predicted_class = torch.argmax(logits, dim=1).item()
    label_map = {0: "in-favor", 1: "against", 2: "neutral-or-unclear"}
    return label_map[predicted_class]

file['label_pred'] = file['clean_tweet'].apply(lambda tweet: get_pred(tweet, tokenizer, model))
file = file[['tweet', 'label_true', 'label_pred']]
file.to_csv(file_path, index=False)

print(f"Accuracy of the FE model: {(file['label_pred'] == file['label_true']).sum() / len(file)}")