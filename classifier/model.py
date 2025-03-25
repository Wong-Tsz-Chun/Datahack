import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import time
status = ['Anxiety', 'Normal', 'Depression', 'Suicidal', 'Stress', 'Bipolar', 'Personality disorder']

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

class Classifier():
    def __init__(self, bert_model_name='bert-base-uncased', model_path='bert_classifier.pth'):
        self.status = ['Anxiety', 'Normal', 'Depression', 'Suicidal', 'Stress', 'Bipolar', 'Personality disorder']
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.model = BERTClassifier(bert_model_name, len(self.status))
        self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), model_path), map_location=torch.device('cpu')))
        self.model.eval()

    def inference(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
        
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
        return self.status[predicted_class]
    
if __name__ == '__main__':
    classifier = Classifier()
    start = time.time()
    print(classifier.inference("trouble sleeping, confused mind, restless heart. All out of tune"))
    end = time.time()
    print(end - start)
