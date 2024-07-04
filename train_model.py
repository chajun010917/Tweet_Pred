import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import T5Tokenizer, TrainingArguments, DataCollatorWithPadding, T5Model 
import re
from custom_T5_model import CustomT5Model, CustomTrainer

#Dataset Cleaning fnct. 
def clean_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)  #Remove URLs
    tweet = re.sub(r'@\w+', '', tweet)   #Remove Mentions
    tweet = re.sub(r'#\w+', '', tweet)   #Remove Hashtags 
    tweet = re.sub(r'\s+', ' ', tweet).strip() #Remove Multiple Spaces  
    tweet = re.sub(r'[^\w\s,]', '', tweet)  #Remove non-alphanumeric characters
    return tweet

#Label Initialization for Feature Extraction fnct.
def normalize_label(label):  
    label = label.lower()
    if "in-favor" in label: return "in-favor"
    elif "neutral-or-unclear" in label: return "neutral-or-unclear"
    elif "against" in label: return "against"
    else: return label 

#Tokenization fnct.
def tokenize_data(examples):
    inputs = tokenizer(examples['cl_tweet'], max_length=512, padding='max_length', truncation=True)
    inputs['labels'] = [0 if label == "in-favor" else 1 if label == "against" else 2 for label in examples['label_true']]
    return inputs

#Augmented Datasets
aug_path = './aug_data.csv' 
aug1_path = './aug_data1.csv'
gai_path = './GAItweets.csv' #Added extra neutral-or-unclear samples due to the model's weakness

aug = pd.read_csv(aug_path)
aug1 = pd.read_csv(aug1_path)
gai = pd.read_csv(gai_path)

gai = gai[gai['label_true'] == "neutral-or-unclear"].sample(n=150, random_state=42)
aug = pd.concat([aug, aug1, gai], ignore_index=True)
aug = aug.sample(frac=1, random_state=32).reset_index(drop=True)

aug['cl_tweet'] = aug['tweet'].apply(clean_tweet)
aug['label_true'] = aug['label_true'].apply(normalize_label)

train_df, test_df = train_test_split(aug, test_size=0.1, stratify=aug['label_true'])

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
dataset_dict = DatasetDict({'train': train_dataset, 'test': test_dataset})

#Model Initialization 
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
base_model = T5Model.from_pretrained("google/flan-t5-large")

for param in base_model.parameters():
    param.requires_grad = False 
    
model = CustomT5Model(base_model)
tokenized_datasets = dataset_dict.map(tokenize_data, batched=True)
data_collator = DataCollatorWithPadding(tokenizer)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=1e-3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    weight_decay=0.01,
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.evaluate()
trainer.save_model('./model_fe')