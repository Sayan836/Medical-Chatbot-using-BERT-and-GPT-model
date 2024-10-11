import torch
from transformers import AutoModel, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import re
import torchbearer
from torchbearer import Trial
from torchbearer.callbacks import EarlyStopping
import utils

device='cuda' if torch.cuda.is_available() else 'cpu'

class Data(torch.utils.data.Dataset):
  def __init__(self, df, tokenizer, max_length):
    self.df = df
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.df)

  def convert_tensor(self, inputs):
    return self.tokenizer(inputs, return_tensors='pt', padding='max_length', max_length=self.max_length, truncation=True)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]
    ques = self.convert_tensor(row['question'])
    ques_ids,ques_mask= ques['input_ids'], ques['attention_mask']
    ans = self.convert_tensor(row['answer'])
    ans_ids, ans_mask= ans['input_ids'], ans['attention_mask']
    target = torch.tensor(row['label'], dtype=torch.float)
    #sample = {'ques_ids': ques_ids, 'ques_mask': ques_mask, 'ans_ids': ans_ids, 'ans_mask': ans_mask, 'target': target}
    return (ques_ids,ques_mask,ans_ids,ans_mask, target)


