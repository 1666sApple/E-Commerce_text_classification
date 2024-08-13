import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import DEVICE, EPOCHS, LEARNING_RATE, NUM_CLASSES
from model.utils import set_seed, load_data, prepare_data, create_data_loaders
from model.dataset import EcommerceReviewDataset
from model.model import SentimentClassifierBert
from model.train import train_model
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    
    # Load and prepare data
    df = load_data("data/ecommerceDataset.csv")
    ids, masks, labels = prepare_data(df, tokenizer)
    dataset = EcommerceReviewDataset(ids, masks, labels)
    train_loader, test_loader = create_data_loaders(dataset)

    # Initialize the model
    model = SentimentClassifierBert(NUM_CLASSES).to(DEVICE)

    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Train the model
    train_loss, test_loss, train_acc, test_acc = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, EPOCHS)

    # Save the model weights
    torch.save(model.state_dict(), 'app/models-weight/sentiment_BERT_ecommerce-review_pytorch.pth')

if __name__ == "__main__":
    main()
