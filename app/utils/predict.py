import torch
from app.models.bert_classifier import SentimentClassifierBert
from model.config import DEVICE, NUM_CLASSES
from transformers import BertTokenizer

model = SentimentClassifierBert(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load('app/models-weight/sentiment_BERT_ecommerce-review_pytorch.pth', map_location=DEVICE))
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def predict_sentiment(text):
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(DEVICE)
    attention_mask = encoded['attention_mask'].to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item()