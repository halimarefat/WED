import numpy as np
import torch
import time
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from tqdm import tqdm
import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import OFLESDataset, R2Score, trainDataCollecter, MOTHERDIR, HEADERS, M1_HEADERS, M4_HEADERS, M3_HEADERS, M2_HEADERS, M6_HEADERS
from model.wae import WAE
from model.mlp import mlp
from utils.loss import WaveletLoss

modelMode = 'MLP' #'WAE' # 
Re = 'R4'
groupName = f'wae_R10{Re[1]}' if modelMode == 'WAE' else f'mlp_R10{Re[1]}'
dt_names = ['M1', 'M2', 'M3', 'M4']

test_org, test_norm, test_means, test_scales = trainDataCollecter(Re)

print(f'For Re = 10^{Re[-1]}:')
for dt_name in dt_names:
    print(f'Working on {dt_name}!')
    dt = test_norm.filter(globals()[f"{dt_name}_HEADERS"], axis=1)
    ds = OFLESDataset(dt)
    test_loader = torch.utils.data.DataLoader(dataset=ds, batch_size=50000, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    PATH = f"{MOTHERDIR}/checkpoints/{groupName}_model_{dt_name}.pt"
    out_channels = 1
    in_channels = dt.shape[1] - out_channels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if modelMode == 'WAE':
        model = WAE(in_channels=in_channels, out_channels=out_channels, bilinear=True)  
    elif modelMode == 'MLP':
        model = mlp(input_size=in_channels, output_size=out_channels, hidden_layers=5, neurons_per_layer=[60,60,60,60,60])  
    model.load_state_dict(torch.load(PATH))
    model.eval()
    model.to(device)
    model.double()
    criterion = nn.MSELoss()

    Cs_true = []
    Cs_pred = []
        
    start_time = time.time()
    test_loss = 0.0
    test_loop = tqdm(test_loader, position=0, leave=True)
    for batch in test_loop:
        features = batch[:, :-1].to(device)
        label = batch[:, -1].to(device)
        output = model(features)
        pred = output.squeeze()
        loss = criterion(pred, label)
        test_loss += loss.item()
        test_loop.set_postfix(loss=loss.item())
        Cs_pred.append(pred.detach().cpu().numpy() * test_scales['Cs'].values + test_means['Cs'].values)
        Cs_true.append(label.detach().cpu().numpy() * test_scales['Cs'].values + test_means['Cs'].values)

    Cs_true = np.concatenate(Cs_true).ravel()
    Cs_pred = np.concatenate(Cs_pred).ravel()

    test_loss /= len(test_loader)
    test_coefficient = R2Score(label, pred).item()

    print(f'loss is {test_loss}, and R2 score is {test_coefficient}')
    print(f'shape of Cs_true is {Cs_true.shape}')
    print(f'shape of Cs_pred is {Cs_pred.shape}')

    n, xedges, yedges = np.histogram2d(Cs_true, Cs_pred, bins=[1500, 1501])
    jpdf = n / trapz(trapz(n, xedges[:-1], axis=0), yedges[:-1])
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

    dir = Path(f'{MOTHERDIR}/Results/{groupName}/')
    if not dir.exists():
        dir.mkdir(parents=True, exist_ok=True)
        
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    
    plt.pcolormesh(X, Y, jpdf.T, shading='auto', cmap='jet')
    plt.clim([-4, 54])
    plt.xlabel(r'$C_s$', fontsize=14)
    plt.ylabel(r'$\tilde{C_s}$', fontsize=14)
    plt.xlim([-0.15, 0.15])
    plt.ylim([-0.15, 0.15])
    plt.xticks(fontsize=18)  
    plt.yticks(fontsize=18)  
    plt.tight_layout()
    plt.savefig(dir / f'{groupName}_{dt_name}_jpdf.png')
    plt.close()
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    plt.hist(Cs_true, bins=1000, density=True, alpha=0.6, histtype=u'step', color='blue')
    plt.hist(Cs_pred, bins=1000, density=True, alpha=0.6, histtype=u'step', color='red')
    plt.xlim([-0.15, 0.15])
    plt.ylim([0, 85])
    plt.xlabel(' ', fontsize=18)
    plt.ylabel(r'density', fontsize=18)
    plt.xticks(fontsize=18)  
    plt.yticks(fontsize=18)
    if modelMode == 'WAE':
        plt.legend([r'$C_s$ (GT)', r'$\hat{C}_s$ (WED)'], fontsize=18, frameon=False)
    elif modelMode == 'MLP':
        plt.legend([r'$C_s$ (GT)', r'$\hat{C}_s$ (ANN)'], fontsize=18, frameon=False)
    plt.savefig(dir / f'{groupName}_{dt_name}_density.png')
    plt.close()
