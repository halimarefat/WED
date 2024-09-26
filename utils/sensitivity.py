import numpy as np
import matplotlib.pyplot as plt
from .utils import FEATNAMES

def sensitivity_analysis(model, X, epsilon=1e-4):
    model.eval()
    model.float()
    sensitivities = np.zeros(X.shape[1])

    for i in range(X.shape[1]):
        original_pred = model(X).detach().cpu().numpy()  # Move tensor to CPU before converting to NumPy
        perturbed_X = X.clone().detach()
        perturbed_X[:, i] += epsilon
        perturbed_pred = model(perturbed_X).detach().cpu().numpy()  # Move tensor to CPU before converting to NumPy
        sensitivities[i] = np.mean(np.abs(perturbed_pred - original_pred))

    max_sensitivity = np.max(sensitivities)
    if max_sensitivity > 0:
        sensitivities_percentage = (sensitivities / max_sensitivity) * 100
    else:
        sensitivities_percentage = sensitivities

    return sensitivities_percentage


def plot_sensitivities(features_name, sensitivities, hatches, labels, path):
    plt.figure(figsize=(15, 5))
    plt.rcParams.update({
                "text.usetex": True,
                "font.family": "Helvetica"
            })
    n_1 = len(features_name[0])
    n_2 = len(features_name[1])

    total_features = n_1 + n_2

    bar_width = 0.55

    indices_1 = np.arange(n_1)
    indices_2 = np.arange(n_1, total_features) - 0.5
    positions_1 = indices_1 - bar_width / 2
    positions_2 = indices_2 + bar_width / 2

    plt.bar(positions_1, sensitivities[0], width=bar_width, color='white', edgecolor='black', hatch=hatches[0], label=labels[0])
    plt.bar(positions_2, sensitivities[1], width=bar_width, color='white', edgecolor='black', hatch=hatches[1], label=labels[1])
    
    mean_sensitivity_1 = np.mean(sensitivities[0])
    mean_sensitivity_2 = np.mean(sensitivities[1])
    #plt.axvline(mean_sensitivity_1, color=colors[0], linestyle='--')
    #plt.axvline(mean_sensitivity_2, color=colors[1], linestyle='--')
    xlab = []
    for i in range(n_1):
        xlab.append(FEATNAMES[features_name[0][i]])

    for i in range(n_2):
        xlab.append(FEATNAMES[features_name[1][i]])
       
    plt.xticks(np.arange(total_features), xlab, rotation=45, ha='right', fontsize=18) 
    plt.yticks(fontsize=20)
    plt.ylabel(r'Sensitivity $[\%]$', fontsize=16) 
    #plt.yticks(np.arange(total_features), ylab, fontsize=16)
    #plt.xlabel(r'Sensitivity $[\%]$', fontsize=16)
    plt.legend(fontsize=18, frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(path)
    plt.close()
    
def bland_altman_plot(y_true, y_pred, path):
    mean_ = np.mean([y_pred, y_true], axis=0)
    diff = y_pred - y_true  

    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    print(f'mean_diff is {mean_diff}')
    
    plt.figure(figsize=(9, 6))
    plt.rcParams.update({
                "text.usetex": True,
                "font.family": "Helvetica"
            })
    plt.scatter(mean_, diff, alpha=0.5)
    plt.axhline(mean_diff, color='red', linestyle='--')
    plt.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--')
    plt.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--')
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.xticks(fontsize=20)  
    plt.yticks(fontsize=20)  
    plt.xlabel(r'$\mu$', fontsize=20)
    plt.ylabel(r'$\Delta$', fontsize=20)
    plt.savefig(path)
    plt.close()
