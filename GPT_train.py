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
from GPT2.gpt_DataLoader import DataLoader

device='cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 does not have a pad token, so using EOS as padding
model = GPT2LMHeadModel.from_pretrained('gpt2')
model = model.to(device)

df= pd.read_csv("Data/GPT_data.csv")

# Step 2: Prepare the data (assuming the DataLoader class has been defined earlier)
train_dataset = DataLoader(df, tokenizer, max_length=1024)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)

# Step 3: Define the optimizer and the loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Step 4: Training Loop
epochs = 10
model.train()

for epoch in range(epochs):
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
    
    for batch in progress_bar:
        # Move the input data to the appropriate device (GPU or CPU)
        input_ids = batch['context'].to(device)
        labels = batch['labels'].to(device)
        loss_mask = batch['loss_mask'].to(device)

        # Forward pass: Get the model's outputs (logits) and compute loss
        outputs = model(input_ids=input_ids, labels=labels)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')  # No reduction yet, loss will be masked
        
        # Compute the loss only for the masked positions (where loss_mask == 1)
        shift_logits = logits[..., :-1, :].contiguous()  # Shift for GPT2 CLM
        shift_labels = labels[..., 1:].contiguous()      # Shifted labels
        active_loss = loss_mask[..., 1:].bool()          # Mask for loss
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        masked_loss = loss.view(shift_labels.size())[active_loss].mean()

        # Backpropagation and optimization step
        optimizer.zero_grad()
        masked_loss.backward()
        optimizer.step()

        # Update the total loss for the progress bar
        total_loss += masked_loss.item()
        progress_bar.set_postfix(loss=total_loss / (len(progress_bar) + 1))
        torch.cuda.empty_cache()
    print(f"Epoch {epoch+1} finished with average loss: {total_loss / len(train_dataloader)}")
utils.Model_Save(model,"chatbot_model","Trained_models/")