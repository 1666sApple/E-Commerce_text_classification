# model/dataset.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
stop_words = set(stopwords.words('english'))

def tokenize(sentence):
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [word for word in word_tokens if word.lower() not in stop_words]
    filtered_sentence = ' '.join(filtered_sentence)
    tokens = tokenizer.encode_plus(
        filtered_sentence,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False,
        return_tensors='pt'
    )
    return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze()

class EcommerceReviewDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }
