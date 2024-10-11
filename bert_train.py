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
from Bert.data_preprocessor import data_preprocessing
from Bert.Dataset import Data
from Bert.model import EarlyStoppingCallback, Feature_Extractor

device='cuda' if torch.cuda.is_available() else 'cpu'

doc= utils.load_dataset()

preprocessor= data_preprocessing(doc)
doc= preprocessor.process()

model_name = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

for name, param in model.named_parameters():
    if "pooler.dense" not in name:
        param.requires_grad = False

max_seq_length = 512
dataset = Data(doc, tokenizer, max_seq_length)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

bert_model = Feature_Extractor(model).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(bert_model.parameters(), lr=0.001)


bert_model.train()
num_epochs = 10

#CallBack function
early_stopping_callback = EarlyStoppingCallback(monitor='loss', patience=2, mode='min')

for epoch in range(1, num_epochs + 1):
  total_correct = 0
  total_samples = 0
  for batch_idx, (ques_ids,ques_mask,ans_ids,ans_mask, target) in enumerate(train_loader):
    # Move tensors to the device
    ques_ids, ques_mask, ans_ids, ans_mask, target = (
        ques_ids.to(device),
        ques_mask.to(device),
        ans_ids.to(device),
        ans_mask.to(device),
        target.to(device),
    )

    optimizer.zero_grad()

    Inputs = torch.stack([ques_ids, ques_mask, ans_ids, ans_mask])
    output = bert_model(Inputs)

    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    if output[0]>=0:
      output[0]=1
    else:
      output[0]=-1
    total_correct += (output == target).sum().item()
    total_samples += target.size(0)
    accuracy=total_correct/total_samples

  state={
    torchbearer.METRICS: {'loss': loss.item()},
    torchbearer.STOP_TRAINING: False
  }
  state=early_stopping_callback.on_end_epoch(state)

  if state[torchbearer.STOP_TRAINING]:
        print("Early stopping triggered.")
        break

  print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}, Accuracy: {accuracy}")

utils.Model_Save(bert_model,"Feature_Extractor","Trained_models/")


