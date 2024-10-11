import torch
import ast
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

device='cuda' if torch.cuda.is_available() else 'cpu'

class DataLoader(torch.utils.data.Dataset):
  def __init__(self, df, tokenizer, max_length):
    self.df = df
    self.tokenizer = tokenizer
    self.max_length = max_length
  
  def __len__(self):
    return len(self.df)
  
  def convert_tensor(self,vec):
    vec= ast.literal_eval(vec)
    n=len(vec)
    if n<1024:
      vec.extend([0]*(1024-n))
    return torch.tensor(vec, dtype=torch.long)

  def __getitem__(self,idx):
    row= self.df.iloc[idx]
    input_ids= self.convert_tensor(row['gpt_data'])
    context = input_ids.clone()
    labels = input_ids.clone()
    loss_mask= self.convert_tensor(row['mask'])
    labels= torch.roll(labels,shifts=-1, dims=0)

    attention_mask = torch.ones_like(input_ids)
    attention_mask[input_ids == self.tokenizer.pad_token_id] = 0

    return {
      'context':context,
      'labels': labels,
      'loss_mask': loss_mask
    }
    
