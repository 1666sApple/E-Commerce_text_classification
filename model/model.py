import torch.nn as nn
from transformers import BertModel

class SentimentClassifierBert(nn.Module):
    def __init__(self, num_classes):
        super(SentimentClassifierBert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(p=0.25)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # Unfreeze the last 6 layers of BERT
        for name, param in self.bert.named_parameters():
            if "encoder.layer" in name:
                layer_num = int(name.split(".")[2])
                if layer_num >= 6:  # Unfreeze layers 6-11 (last 6 layers)
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits