import torch
import time
import json
import os
import sys
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

from utils.utils import OFLESDataset, R2Score, trainDataCollecter, MOTHERDIR
from utils.utils import OFLESDataset, R2Score, M1_HEADERS, M2_HEADERS, M3_HEADERS, M4_HEADERS, M5_HEADERS
from model.wae import WAE
from model.mlp import mlp
from utils.loss import WaveletLoss

wavelet = False
modelMode = 'WAE' # 'MLP' #
Re = 'R4'
Mconf = '3'
groupName = f'wae_R10{Re[1]}' if modelMode == 'WAE' else f'mlp_R10{Re[1]}'
dt_name = f'M{Mconf}_wavelet' if wavelet else f'M{Mconf}'

train_org, train_norm, train_means, train_scales = trainDataCollecter(Re)
dt = train_norm.filter(globals()[f"{dt_name[:2]}_HEADERS"], axis=1)

learning_rate = 0.001
num_epochs = 500
patience = 60
best_model_path = f'{MOTHERDIR}/checkpoints/{groupName}_model_{dt_name}.pt'
out_channels = 1
in_channels = dt.shape[1] - out_channels 
split_sz = 0.8
batch_sz_trn = 4096
batch_sz_val = int(batch_sz_trn / 4)

mask = np.random.rand(len(dt)) < split_sz
train = dt[mask].reset_index(drop=True) 
val = dt[~mask].reset_index(drop=True)

train_dataset = OFLESDataset(train)
val_dataset = OFLESDataset(val)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_sz_trn, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_sz_val, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if modelMode == 'WAE':
    model = WAE(in_channels=in_channels, out_channels=out_channels, bilinear=True)  
elif modelMode == 'MLP':
    model = mlp(input_size=in_channels, output_size=out_channels, hidden_layers=5, neurons_per_layer=[60,60,60,60,60])  
    
model.to(device)
model.double()
criterion = WaveletLoss(wavelet='db1') if wavelet else nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=3, gamma=0.2)

history = {
    "train_loss": [],
    "val_loss": [],
    "train_coefficient": [],
    "val_coefficient": [],
    "learning_rates": [],
    "epoch_times": []
}

best_val_loss = np.inf
patience_counter = 0

log_file = f"./logs/{groupName}_training_log_{dt_name}.txt"

for epoch in range(num_epochs):
    start_time = time.time()
    
    model.train()
    train_loss = 0.0
    y_train_true = []
    y_train_pred = []
    
    train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Training", leave=False)
    for batch in train_loop:
        inputs = batch[:, 0:-1].to(device)
        target = batch[:, -1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        y_train_true.append(target)
        y_train_pred.append(outputs.squeeze())
        
        train_loop.set_postfix(loss=loss.item())
    
    train_loss /= len(train_loader)
    y_train_true = torch.cat(y_train_true)
    y_train_pred = torch.cat(y_train_pred)
    train_coefficient = R2Score(y_train_true, y_train_pred).item()
    
    model.eval()
    val_loss = 0.0
    y_val_true = []
    y_val_pred = []
    
    val_loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Validation", leave=False)
    with torch.no_grad():
        for batch in val_loop:
            inputs = batch[:, 0:-1].to(device)
            target = batch[:, -1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), target)
            val_loss += loss.item()
            y_val_true.append(target)
            y_val_pred.append(outputs.squeeze())
            
            val_loop.set_postfix(loss=loss.item())
    
    val_loss /= len(val_loader)
    y_val_true = torch.cat(y_val_true)
    y_val_pred = torch.cat(y_val_pred)
    val_coefficient = R2Score(y_val_true, y_val_pred).item()
    
    scheduler.step()
    epoch_duration = time.time() - start_time
    
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_coefficient"].append(train_coefficient)
    history["val_coefficient"].append(val_coefficient)
    history["learning_rates"].append(optimizer.param_groups[0]['lr'])
    history["epoch_times"].append(epoch_duration)
    
    log_message = (f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Train R^2: {train_coefficient:.4f}, "
                   f"Val R^2: {val_coefficient:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}, "
                   f"Time: {epoch_duration:.2f}s\n")
    
    with open(log_file, 'a') as f:
        f.write(log_message)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"{log_message.strip()} -> Saving best model")
    else:
        patience_counter += 1
        print(f"{log_message.strip()} -> No improvement")
        
    if patience_counter >= patience:
        print("Early stopping triggered")
        break

print(f"Training complete. \n Best model saved to '{best_model_path}'.")

with open(f"./logs/{groupName}_training_history_{dt_name}.json", "w") as f:
    json.dump(history, f)
    
print(f"\n Training history saved to '{MOTHERDIR}/logs/{groupName}_training_history_{dt_name}.json'")

data_iter = iter(train_loader)
next(data_iter)[:,0:-1]

traced_script_module = torch.jit.trace(model, next(data_iter)[:,0:-1].to(device))
traced_script_module.save(f"./traced/{groupName}_traced_model_{dt_name}.pt")

print(f"Traced model saved to '{MOTHERDIR}/traced/{groupName}_traced_model_{dt_name}.pt'")
