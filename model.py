import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import RobertaTokenizer, T5ForConditionalGeneration

class MarkdownModel(nn.Module):
    def __init__(self, model_path):
        super(MarkdownModel, self).__init__()
        #self.model = AutoModel.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base').encoder
        
        #simple_dropout
        self.dropout = nn.Dropout(0.2)
        
        #multi-sample-drop-out
        #self.dropout1 = nn.Dropout(0.1)
        #self.dropout2 = nn.Dropout(0.2)
        #self.dropout3 = nn.Dropout(0.3)
        #self.dropout4 = nn.Dropout(0.4)
        #self.dropout5 = nn.Dropout(0.5)
        
        self.top = nn.Linear(769, 1)
        
        self.model.embed_tokens.requires_grad_(False)
        #self.model.embeddings.requires_grad_(False)
        self.model.block[:6].requires_grad_(False) 
        #self.model.layer[:6].requires_grad_(False) 
            

    def forward(self, ids, mask, fts):
        
        emb = self.model(ids, mask).last_hidden_state
        
        mean_emb = torch.mean(emb, 1)
        logits = torch.cat((mean_emb, fts), 1) 

        #preds1 = self.top(self.dropout1(logits))
        #preds2 = self.top(self.dropout2(logits))
        #preds3 = self.top(self.dropout3(logits))
        #preds4 = self.top(self.dropout4(logits))
        #preds5 = self.top(self.dropout5(logits))

        #multi-drop-out
        #preds = (preds1 + preds2 + preds3 + preds4 + preds5) / 5
        
        preds = self.top(self.dropout(logits))
        
        return preds
