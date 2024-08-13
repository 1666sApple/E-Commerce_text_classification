import torch

SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 2e-5
MAX_LENGTH = 512

# To determine NUM_CLASSES, we need to analyze the dataset
import pandas as pd

def get_num_classes():
    df = pd.read_csv("data/ecommerceDataset.csv", header=None, names=['category', 'text'])
    return len(df['category'].unique())

NUM_CLASSES = get_num_classes()
print(f"Number of classes: {NUM_CLASSES}")