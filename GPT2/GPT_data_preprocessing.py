import torch
import faiss
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

class GPT_DataPrepocessor():
  def __init__(self, data, tokenizer,max_length=1024, topk=20):
    self.data= data
    self.tokenizer= tokenizer
    self.topk= topk
    self.max_length = max_length
    self.special_token = self.tokenizer.encode('answer')[0]
  
  def extract_embedding(self):
    question_bert= self.data['question_embedding'].tolist()
    answer_bert= self.data['answer_embedding'].tolist()
    question_bert= np.array(question_bert).astype('float32')
    answer_bert= np.array(answer_bert).astype('float32')
    return answer_bert, question_bert
  
  def create_index(self,answer_bert):
    answer_bert= np.mean(answer_bert, axis=2).squeeze()
    answer_index= faiss.IndexFlatIP(answer_bert.shape[-1])
    answer_index.add(answer_bert)
    return answer_index

  def get_topk_answers(self,index,question_embedding):
  # Search for the top-k similar answer embeddings
    scores, indices = index.search(question_embedding.reshape(1, -1), self.topk)
    return indices[0], scores[0]

  def format_embedding(self,vec):
    vec= np.mean(vec,axis=1)
    return vec
  
  def mask_start(self, gpt_data):
    return 1024-gpt_data[::-1].index(4600)+1

  def return_loss_mask(self, mask_start):
    return [0]*mask_start+[1]*(1024-mask_start)
  
  def preparing_gpt_training_data(self,question,answer,question_embedding,answer_index):
    topk=20
    scores,indices=answer_index.search(self.format_embedding(question_embedding), topk)
    q_sub=self.data.iloc[indices.reshape(20)]
  
    line = '`QUESTION: %s `ANSWER: %s' % (
                        question, answer)
    encoded_len=len(tokenizer.encode(line))
    for i in q_sub.iterrows():
      line='`QUESTION: %s `ANSWER: %s ' % (i[1]['question'],i[1]['answer']) + line
      line=line.replace('\n','')
      encoded_len=len(tokenizer.encode(line))
      if encoded_len>=1024:
        break
    return tokenizer.encode(line)[-1024:]
  
  def process(self):
    tqdm.pandas()
    answer_bert, question_bert= self.extract_embedding()
    answer_index= self.create_index(answer_bert)
    print("FAISS Index created successfully.....")
    entries=[]
    for i in range(len(self.data)):
      line=self.preparing_gpt_training_data(self.data['answer'][i], self.data['question'][i], question_bert[i],answer_index)
      entries.append(line)
    self.data['gpt_data']=entries

    self.data['mask_start'] = self.data.gpt_data.progress_apply(lambda x: self.mask_start(x))

        # Find the length of the GPT-2 encoded data
    self.data['gpt_lens'] = self.data.gpt_data.apply(lambda x: len(x))

        # Filter data to only include sequences of length 1024
    print("length of data before filtering: ",len(self.data))
    print(self.data.head())
    gpt_data_cleaned = self.data[self.data.gpt_lens == self.max_length]
    print("length of data after filtering:", len(gpt_data_cleaned))

        # Create loss masks
    gpt_data_cleaned['mask'] = gpt_data_cleaned.mask_start.apply(lambda x: self.return_loss_mask(x))
    gpt_data_cleaned.to_csv("Data/GPT_data.csv")
    print("Data created and saved successfully....")


data= torch.load("Data/QA_with_embeddings.pt", map_location=torch.device('cpu'))
data= pd.DataFrame(data)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
preprocessor= GPT_DataPrepocessor(data,tokenizer)
preprocessor.process()



    
    

