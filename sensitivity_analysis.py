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
from utils.utils import OFLESDataset, R2Score, trainDataCollecter, MOTHERDIR, HEADERS, M1_HEADERS, M4_HEADERS, M3_HEADERS, M2_HEADERS, M6_HEADERS, M5_HEADERS
from model.wae import WAE
from model.mlp import mlp
from utils.loss import WaveletLoss
from utils.sensitivity import sensitivity_analysis, plot_sensitivities, bland_altman_plot

for mmode in ['WAE']:
    modelMode = mmode #'MLP' #'WAE' # 
    for re in ['R3', 'R4']: #'R4']: #
        print(f'for {mmode} and {re}:')
        Re = re #'R3'
        groupName = f'wae_R10{Re[1]}' if modelMode == 'WAE' else f'mlp_R10{Re[1]}'
        dt_names = [['M1', 'M3'], ['M2', 'M4']] #['M2', 'M6']] #

        test_org, test_norm, test_means, test_scales = trainDataCollecter(Re)

        print(f'For Re = 10^{Re[-1]}:')
        for dt_name in dt_names:
            print(f'Working on {dt_name}!')
            out_channels = 1
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            dt_1 = test_norm.filter(globals()[f"{dt_name[0]}_HEADERS"], axis=1)
            ds_1 = OFLESDataset(dt_1)
            test_loader_1 = torch.utils.data.DataLoader(dataset=ds_1, batch_size=50000, shuffle=False)
            PATH_1 = f"{MOTHERDIR}/checkpoints/{groupName}_model_{dt_name[0]}.pt"
            in_channels_1 = dt_1.shape[1] - out_channels
            if modelMode == 'WAE':
                model_1 = WAE(in_channels=in_channels_1, out_channels=out_channels, bilinear=True) 
            elif modelMode == 'MLP':
                model_1 = mlp(input_size=in_channels_1, output_size=out_channels, hidden_layers=5, neurons_per_layer=[60,60,60,60,60])  
            model_1.load_state_dict(torch.load(PATH_1))
            model_1.eval()
            model_1.to(device)
            model_1 = model_1.float()
            
            dt_2 = test_norm.filter(globals()[f"{dt_name[1]}_HEADERS"], axis=1)
            ds_2 = OFLESDataset(dt_2)
            test_loader_2 = torch.utils.data.DataLoader(dataset=ds_2, batch_size=50000, shuffle=False)
            PATH_2 = f"{MOTHERDIR}/checkpoints/{groupName}_model_{dt_name[1]}.pt"
            in_channels_2 = dt_2.shape[1] - out_channels
            if modelMode == 'WAE':
                model_2 = WAE(in_channels=in_channels_2, out_channels=out_channels, bilinear=True) 
            elif modelMode == 'MLP':
                model_2 = mlp(input_size=in_channels_2, output_size=out_channels, hidden_layers=5, neurons_per_layer=[60,60,60,60,60])
            model_2.load_state_dict(torch.load(PATH_2))
            model_2.eval()
            model_2.to(device)
            model_2 = model_2.float()
            
            criterion = nn.MSELoss()

            test_loop_1 = tqdm(test_loader_1, position=0, leave=True)
            test_loop_2 = tqdm(test_loader_2, position=0, leave=True)
            
            Cs_true_1, Cs_pred_1 = [], []
            Cs_true_2, Cs_pred_2 = [], []
            
            for batch_1, batch_2 in zip(test_loop_1, test_loop_2):
                features_1 = batch_1[:, :-1].to(device).float()
                label_1 = batch_1[:, -1].to(device).float()
                output_1 = model_1(features_1)
                pred_1 = output_1.squeeze()
                Cs_pred_1.append(pred_1.detach().cpu().numpy() * test_scales['Cs'].values + test_means['Cs'].values)
                Cs_true_1.append(label_1.detach().cpu().numpy() * test_scales['Cs'].values + test_means['Cs'].values)

                features_2 = batch_2[:, :-1].to(device).float()
                label_2 = batch_2[:, -1].to(device).float()
                output_2 = model_2(features_2)
                pred_2 = output_2.squeeze()
                Cs_pred_2.append(pred_2.detach().cpu().numpy() * test_scales['Cs'].values + test_means['Cs'].values)
                Cs_true_2.append(label_2.detach().cpu().numpy() * test_scales['Cs'].values + test_means['Cs'].values)
                
                sensitivities_1 = sensitivity_analysis(model_1, features_1)
                sensitivities_2 = sensitivity_analysis(model_2, features_2)

                feature_names_1 = globals()[f"{dt_name[0]}_HEADERS"][:-1]  
                feature_names_2 = globals()[f"{dt_name[1]}_HEADERS"][:-1]  
                #colors = ['cyan', 'magenta'] if dt_name in [['M1', 'M3']] else ['red', 'blue']
                hatches = ['/', 'x'] if dt_name in [['M1', 'M3']] else ['\\', 'o']
                labels = ['C1', 'C3'] if dt_name in [['M1', 'M3']] else ['C2', 'C4']
                plot_sensitivities([feature_names_1, feature_names_2], [sensitivities_1, sensitivities_2], hatches, labels, Path(f'{MOTHERDIR}/Results/{groupName}/{dt_name}_sensitivity.png'))
                break  
            
            Cs_true_1 = np.concatenate(Cs_true_1).ravel()
            Cs_pred_1 = np.concatenate(Cs_pred_1).ravel()
            Cs_true_2 = np.concatenate(Cs_true_2).ravel()
            Cs_pred_2 = np.concatenate(Cs_pred_2).ravel()

        
            bland_altman_plot(Cs_true_1, Cs_pred_1, Path(f'{MOTHERDIR}/Results/{groupName}/{dt_name[0]}_bland_altman.png'))
            bland_altman_plot(Cs_true_2, Cs_pred_2, Path(f'{MOTHERDIR}/Results/{groupName}/{dt_name[1]}_bland_altman.png'))
