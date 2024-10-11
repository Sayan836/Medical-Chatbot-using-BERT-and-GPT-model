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

device='cuda' if torch.cuda.is_available() else 'cpu'

def extract_negative_samples(question,tags,doc):
  stop=False
  while (not stop):
    sample_row= doc.sample()
    sample_tags=sample_row.tags.values[0]
    inter_tags=set(tags[0]).intersection(set(sample_tags))
    if len(inter_tags)==0:
      stop=True
  return sample_row

def decontractions(phrase):
    """decontracted takes text and convert contractions into natural form.
     ref: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/47091490#47091490"""
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"won\’t", "will not", phrase)
    phrase = re.sub(r"can\’t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    phrase = re.sub(r"n\’t", " not", phrase)
    phrase = re.sub(r"\’re", " are", phrase)
    phrase = re.sub(r"\’s", " is", phrase)
    phrase = re.sub(r"\’d", " would", phrase)
    phrase = re.sub(r"\’ll", " will", phrase)
    phrase = re.sub(r"\’t", " not", phrase)
    phrase = re.sub(r"\’ve", " have", phrase)
    phrase = re.sub(r"\’m", " am", phrase)

    return phrase


def preprocess(text):
    text = text.lower()
    text = decontractions(text)
    text = re.sub('[$)\?"’.°!;\'€%:,(/]', '', text)
    text = re.sub('\u200b', ' ', text)
    text = re.sub('\xa0', ' ', text)
    text = re.sub('-', ' ', text)
    return text

def load_dataset(embedding_data=False):
  if embedding_data:
    doc=pd.read_csv("/content/drive/MyDrive/Projects/DocProduct/Trained_models/refined_df.csv")
    return doc
  doc1=pd.read_json("Data/ehealthforumQAs.json")
  doc2=pd.read_json("Data/icliniqQAs.json")

  doc1=doc1.drop(['url'],axis=1)
  doc2=doc2.drop(['url','question_text'],axis=1)
  doc=pd.concat([doc1,doc2],axis=0)
  doc=doc.reset_index(drop=True)
  return doc
 

def Model_Save(trained_model,name,path):
  saved_model=torch.save(trained_model,path+name+".pth")
  print("Model saved successfully...")
  return saved_model

def Model_Load(model_path):
  loaded_model=torch.load(model_path)
  return loaded_model

def save_df(df,path):
  df.to_csv(path)
