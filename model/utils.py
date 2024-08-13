import random
import numpy as np
import torch
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import random_split, DataLoader
from model.dataset import EcommerceReviewDataset
from model.config import MAX_LENGTH, BATCH_SIZE

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_data(file_path):
    df = pd.read_csv(file_path, header=None, names=['category', 'text'])
    df.dropna(axis=0, inplace=True)
    return df

def prepare_data(df, tokenizer):
    ids = torch.zeros(((len(df), MAX_LENGTH)), dtype=torch.long)
    masks = torch.zeros(((len(df), MAX_LENGTH)), dtype=torch.long)

    for i, txt in enumerate(df['text']):
        encoded = tokenizer.encode_plus(
            txt,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        ids[i] = encoded['input_ids'].squeeze()
        masks[i] = encoded['attention_mask'].squeeze()

    # Label encoding
    unique_categories = df['category'].unique()
    category_to_index = {category: idx for idx, category in enumerate(unique_categories)}
    labels = torch.tensor([category_to_index[category] for category in df['category']])

    return ids, masks, labels

def create_data_loaders(dataset, train_split=0.75):
    train_size = int(len(dataset) * train_split)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader