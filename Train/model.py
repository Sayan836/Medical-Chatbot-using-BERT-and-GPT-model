import torch
from transformers import AutoModel, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import re
import torchbearer
from torchbearer import Callback
from torchbearer import Trial
from torchbearer.callbacks import EarlyStopping
import utils

device='cuda' if torch.cuda.is_available() else 'cpu'

class EarlyStoppingCallback(Callback):
    def __init__(self, monitor='val_loss', patience=0, mode='auto'):
        super(EarlyStoppingCallback, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best = None
        self.num_bad_epochs = 0

    def on_end_epoch(self, state):
        current = state[torchbearer.METRICS][self.monitor]
        if self.best is None or (self.mode == 'min' and current > self.best) or (self.mode == 'min' and current < self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            state[torchbearer.STOP_TRAINING] = True
        return state

class Feature_Extractor(torch.nn.Module):
    def __init__(self, bert):
        super(Feature_Extractor, self).__init__()
        self.bert = bert

    def FCNN(self, x, input_dim, output_dim):
        fcnn_layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
        ).to(device)
        return fcnn_layer(x)

    def forward(self,Inputs):
        ques_ids, ques_mask, ans_ids, ans_mask = Inputs[0],Inputs[1],Inputs[2],Inputs[3]

        ques_embedding = self.bert(input_ids=ques_ids.view(1,512), attention_mask=ques_mask.view(1,512)).pooler_output
        ans_embedding = self.bert(input_ids=ans_ids.view(1,512), attention_mask=ans_mask.view(1,512)).pooler_output

        ques_fcnn_layer = self.FCNN(ques_embedding, 768, 768)
        ans_fcnn_layer = self.FCNN(ans_embedding, 768, 768)
        
        ques_fcnn_layer= ques_fcnn_layer + ques_embedding
        ans_fcnn_layer= ans_fcnn_layer + ans_embedding

        cos=torch.nn.CosineSimilarity(dim=1,eps=1e-6)
        output = cos(ques_fcnn_layer, ans_fcnn_layer)
        return output