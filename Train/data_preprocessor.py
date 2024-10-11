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

class data_preprocessing():
  def __init__(self,doc):
    self.doc=doc
  
  def neg_data(self):
    #preparing the negative labelled dataset
    tqdm.pandas()
    negative_labels=self.doc.progress_apply(lambda x: pd.Series([x.question,utils.extract_negative_samples(x.question,x.tags,self.doc).answer.values[0],x.tags]),axis=1)
    negative_labels['label']=-1.0
    negative_labels.columns=['question','answer','tags','label']
    self.doc['label']=[1]*636
    self.doc=pd.concat([self.doc,negative_labels],axis=0)
    self.doc=self.doc.reset_index(drop=True)
  
  def preprocessing_text(self,questions,answers):
    tokenized_ques=[]
    tokenized_ans=[]
    for question,answer in zip(questions,answers):
      question=utils.preprocess(question)
      answer=utils.preprocess(answer)
  
  def process(self):
    self.neg_data()
    self.preprocessing_text(self.doc['question'],self.doc['answer'])
    return self.doc
    
