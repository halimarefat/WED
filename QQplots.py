import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from utils.utils import OFLESDataset, R2Score, trainDataCollecter, testDataCollecter, MOTHERDIR, HEADERS, M1_HEADERS, M4_HEADERS, M3_HEADERS, M2_HEADERS, M6_HEADERS
from utils.ice import generate_ice_data, plot_ice, load_model
import scipy.stats as stats

for mmode in ['WAE']: #'MLP', 'WAE']:
    modelMode = mmode #'MLP' #'WAE' # 
    for re in ['R4'] : #'R3', 'R4']:
        print(f'for {mmode} and {re}:')
        Re = re #'R3'
        groupName = f'wae_R10{Re[1]}' if modelMode == 'WAE' else f'mlp_R10{Re[1]}'

        dt_names = ['M6'] #'M1', 'M2', 'M3', 'M4']

        test_org, test_norm, test_means, test_scales = testDataCollecter(Re)
        train_org, train_norm, train_means, train_scales = trainDataCollecter(Re)

        dir = Path(f'{MOTHERDIR}/Results/{groupName}/')
        if not dir.exists():
            dir.mkdir(parents=True, exist_ok=True)

        Num = 45000
        for dt_name in dt_names:
            print(f'Model Config {dt_name}:')
            dt = test_norm.sample(n=Num, random_state=42).reset_index(drop=True)
            dt = dt.filter(globals()[f"{dt_name}_HEADERS"], axis=1)
            ds = OFLESDataset(dt)
            ds_loader = torch.utils.data.DataLoader(dataset=ds, batch_size=Num, shuffle=False)

            feature_names = globals()[f"{dt_name}_HEADERS"]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_path = f'{MOTHERDIR}/checkpoints/{groupName}_model_{dt_name}.pt'
            model = load_model(modelMode, model_path, ds[0].shape[0] - 1, 1, device)

            for batch_idx, batch in enumerate(ds_loader):
                print(f'Processing batch {batch_idx+1}/{len(ds_loader)}')
                features = batch[:, 0:-1].to(device)
                target = batch[:, -1]#.to(device)

                # Model output
                model_output = model(features).detach().cpu().numpy().squeeze()

                sorted_ground_truth = np.sort(target)
                sorted_predictions = np.sort(model_output)

                # Create the Q-Q plot
                plt.figure(figsize=(7, 5.5))
                plt.rcParams.update({
                        "text.usetex": True,
                        "font.family": "Helvetica"
                    })
                plt.plot(sorted_ground_truth, sorted_predictions, 'o', markersize=4, markeredgewidth=1, markeredgecolor='blue', markerfacecolor='none')
                plt.plot([sorted_ground_truth.min(), sorted_ground_truth.max()], [sorted_ground_truth.min(), sorted_ground_truth.max()], 'r--')  # 45-degree line
                plt.xlabel(r'$C_s$', fontsize=18)
                plt.ylabel(r'$\hat{C}_s$', fontsize=18)
                plt.xticks(fontsize=18)  
                plt.yticks(fontsize=18)
                plt.xlim([-2,10])
                plt.ylim([-2,10])
                #plt.title('Q-Q Plot: Ground Truth vs. Predictions')
                #plt.grid(True)
                plt.savefig(dir / f'{groupName}_{dt_name}_ModelOutput_QQ.png')
                plt.close()
                
                break
