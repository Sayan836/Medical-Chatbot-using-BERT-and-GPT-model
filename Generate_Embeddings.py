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
from Train.model import Feature_Extractor

import utils

device='cuda' if torch.cuda.is_available() else 'cpu'

class GenerateEmbedding():
  def __init__(self,bert,doc,tokenizer,device):
    self.bert=bert
    self.tokenizer=tokenizer
    self.doc=doc
    self.device=device

  def tokenize(self,text):
    return self.tokenizer(text,return_tensors='pt',padding='max_length',max_length=512,truncation=True).to(self.device)

  def get_embedding(self,text):
    input=self.tokenize(text)
    x= self.bert(input_ids=input['input_ids'],attention_mask=input['attention_mask']).last_hidden_state
    return x

  def generate(self):
    data=[]
    for i in range(len(self.doc)):
      answer_embedding= self.get_embedding(self.doc['answer'][i])
      question_embedding= self.get_embedding(self.doc['question'][i])
      entry = {
        'question': self.doc['question'][i],
        'answer': self.doc['answer'][i],
        'question_embedding': question_embedding,
        'answer_embedding': answer_embedding
      }
      data.append(entry)
    return data

extractor=torch.load("/content/drive/MyDrive/Projects/DocProduct/Trained_models/Feature_Extractor.pth")
extractor.eval()

doc= utils.load_dataset()


pd.options.mode.copy_on_write = True
model_name = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator=GenerateEmbedding(extractor.bert,doc,tokenizer,device)
data= generator.generate()
torch.save(data,"Data/QA_with_embeddings.pt")

