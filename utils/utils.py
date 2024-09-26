import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset

sys_epsilon = sys.float_info.epsilon

MOTHERDIR = os.path.abspath(os.curdir)

FEATNAMES= {"t": r'$t$',                                            
           "X": r'$x$', "Y": r'$y$', "Z": r'$z$',                                  
           "Ux": r'$u$', "Uy": r'$v$', "Uz": r'$w$',                               
           "G1": r'$\mathcal{G}_{xx}$', "G2": r'$\mathcal{G}_{xy}$', "G3": r'$\mathcal{G}_{xz}$', 
           "G4": r'$\mathcal{G}_{yy}$', "G5": r'$\mathcal{G}_{yz}$', "G6": r'$\mathcal{G}_{zz}$',             
           "S1": r'$\mathcal{S}_{xx}$', "S2": r'$\mathcal{S}_{xy}$', "S3": r'$\mathcal{S}_{xz}$', 
           "S4": r'$\mathcal{S}_{yy}$', "S5": r'$\mathcal{S}_{yz}$', "S6": r'$\mathcal{S}_{zz}$',             
           "UUp1": r'$\tau^{\prime}_{xx}$', "UUp2": r'$\tau^{\prime}_{xy}$', "UUp3": r'$\tau^{\prime}_{xz}$', 
           "UUp4": r'$\tau^{\prime}_{yy}$', "UUp5": r'$\tau^{\prime}_{yz}$', "UUp6": r'$\tau^{\prime}_{zz}$', 
           "Cs": r'$C_s$'}                                          

HEADERS = ["t",                                             # time
           "X", "Y", "Z",                                   # spacial coordinates
           "Ux", "Uy", "Uz",                                # velocity components
           "G1", "G2", "G3", "G4", "G5", "G6",              # velocity gradient tensor components
           "S1", "S2", "S3", "S4", "S5", "S6",              # strain rate tensor compnents
           "UUp1", "UUp2", "UUp3", "UUp4", "UUp5", "UUp6",  # resolved Reynolds stress tensor components
           "Cs"]                                            # Smagorinsky coefficient

M1_HEADERS = ['Ux', 'Uy', 'Uz', 'S1',  'S2', 'S3', 'S4', 'S5', 'S6', 'Cs']
M2_HEADERS = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'S1',  'S2', 'S3', 'S4', 'S5', 'S6', 'Cs']
M3_HEADERS = ['Ux', 'Uy', 'Uz', 'UUp1',  'UUp2', 'UUp3', 'UUp4', 'UUp5', 'UUp6', 'Cs']
M4_HEADERS = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'UUp1',  'UUp2', 'UUp3', 'UUp4', 'UUp5', 'UUp6', 'Cs']
M5_HEADERS = ['Ux', 'Uy', 'Uz', 
              'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 
              'S1',  'S2', 'S3', 'S4', 'S5', 'S6', 
              'UUp1',  'UUp2', 'UUp3', 'UUp4', 'UUp5', 'UUp6', 
              'Cs']
M6_HEADERS = ['t',                                            
              'X', 'Y', 'Z',
              'Ux', 'Uy', 'Uz', 
              'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 
              'S1',  'S2', 'S3', 'S4', 'S5', 'S6', 
              'UUp1',  'UUp2', 'UUp3', 'UUp4', 'UUp5', 'UUp6', 
              'Cs']

class OFLESDataset(Dataset):
    
    def __init__(self, dataframe):
        self.data = dataframe

    def __getitem__(self, index):
        if index < len(self.data):
            return torch.tensor(self.data.iloc[index].values, dtype=torch.float64)
        else:
            raise IndexError(f"Index {index} out of range for dataset with length {len(self.data)}")

    def __len__(self):
        return len(self.data)
    
def R2Score(y_true, y_pred):
    SS_res = torch.sum(torch.square(y_true - y_pred))
    SS_tot = torch.var(y_true, unbiased=False) * y_true.size(0)
    return 1 - SS_res / (SS_tot + sys_epsilon)


def trainDataCollecter(Re):
    with open(f'{MOTHERDIR}/datasets/coeffs/train/fieldData_{Re}_seen_means.txt', 'r') as file:
        data = [float(line.strip()) for line in file]
    means = pd.DataFrame(np.reshape(data, (-1, len(HEADERS))), columns=HEADERS)

    with open(f'{MOTHERDIR}/datasets/coeffs/train/fieldData_{Re}_seen_scales.txt', 'r') as file:
        data = [float(line.strip()) for line in file]
    scales = pd.DataFrame(np.reshape(data, (-1, len(HEADERS))), columns=HEADERS)

    norm = pd.read_csv(f'{MOTHERDIR}/datasets/normalized/train/fieldData_{Re}_seen_norm.txt', sep=' ', names=HEADERS)
    org = pd.read_csv(f'{MOTHERDIR}/datasets/original/train/fieldData_{Re}_seen.txt', sep=' ', names=HEADERS)

    return org, norm, means, scales

def testDataCollecter(Re):
    with open(f'{MOTHERDIR}/datasets/coeffs/test/fieldData_{Re}_unseen_means.txt', 'r') as file:
        data = [float(line.strip()) for line in file]
    means = pd.DataFrame(np.reshape(data, (-1, len(HEADERS))), columns=HEADERS)

    with open(f'{MOTHERDIR}/datasets/coeffs/test/fieldData_{Re}_unseen_scales.txt', 'r') as file:
        data = [float(line.strip()) for line in file]
    scales = pd.DataFrame(np.reshape(data, (-1, len(HEADERS))), columns=HEADERS)

    norm = pd.read_csv(f'{MOTHERDIR}/datasets/normalized/test/fieldData_{Re}_unseen_norm.txt', sep=' ', names=HEADERS)
    org = pd.read_csv(f'{MOTHERDIR}/datasets/original/test/fieldData_{Re}_unseen.txt', sep=' ', names=HEADERS)

    return org, norm, means, scales

if __name__ == '__main__':
    ff = ['t', 'X']
    
    print(FEATNAMES[ff[1]])
