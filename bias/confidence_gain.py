import matplotlib.pyplot as plt
import numpy as np
import os

from refnx.dataset import ReflectDataset
from refnx.analysis import Objective, CurveFitter

from structures import Bilayer
from simulate import simulate
from utils import calc_FIM

def contrast_choice(contrasts, angle_times):
    FIM_sums, fitting_sums = [], []
    for x, contrast in enumerate(contrasts):
        DMPC = Bilayer()
        xi = DMPC.parameters
            
        true_vals = np.asarray([param.value for param in xi])
        
        model, data = simulate(DMPC.using_contrast(contrast), angle_times, include_counts=True)
        q, counts = data[:,0], data[:,3]

        g = 1 / calc_FIM([q], xi, counts, [model])
        
        for param in xi:
            param.bounds = (param.value*0.75, param.value*1.25)
        
        objective = Objective(model, ReflectDataset([data[:,0], data[:,1], data[:,2]]))

        fitter = CurveFitter(objective)
        fitter.fit('differential_evolution', verbose=False)

        fitted_vals = np.asarray([param.value for param in xi])
        cov = objective.covar()
        
        g = (g / true_vals[:, np.newaxis]) / true_vals[np.newaxis, :]
        cov = (cov / fitted_vals[:, np.newaxis]) / fitted_vals[np.newaxis, :]
        
        FIM_sums.append(np.sum(g))
        fitting_sums.append(np.sum(cov))
        
        
        print("{0}/{1}".format(x+1, len(contrasts)))
    
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)
    ax.plot(contrasts, FIM_sums, label='Inverse Fisher Information') 
    ax.plot(contrasts, fitting_sums, label='Fitting Covariance') 
    ax.set_xlabel("$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$", fontsize=11, weight='bold')
    ax.set_ylabel("Normalised Matrix Sum", fontsize=11, weight='bold')
    ax.legend()

def confidence_gain(initial_contrast, new_contrasts, angle_times):
    DMPC = Bilayer()

    model_initial, data_initial = simulate(DMPC.using_contrast(initial_contrast), angle_times, include_counts=True)
    q_initial, counts_initial = data_initial[:,0], data_initial[:,3]
    
    xi = DMPC.parameters
    
    g = calc_FIM([q_initial], xi, counts_initial, [model_initial])
    heights_initial = {}
    for i, param in enumerate(xi):
        heights_initial[param] = np.asarray([ellipse_height(g, i, j) for j in range(len(xi))])
    
    heights_new = {param: [] for param in xi}
    for new_contrast in new_contrasts:
        model_new, data_new = simulate(DMPC.using_contrast(new_contrast), angle_times, include_counts=True)
        q_new, counts_new = data_new[:,0], data_new[:,3]
        
        qs = [q_initial, q_new]
        models = [model_initial, model_new]
        counts = np.concatenate((counts_initial, counts_new))
        
        g = calc_FIM(qs, xi, counts, models)
        
        for i, param in enumerate(xi):
            heights = np.asarray([ellipse_height(g, i, j) for j in range(len(xi))])
            heights_new[param].append(heights_initial[param] - heights)
            
    plot_confidences(heights_new, new_contrasts)     

def ellipse_height(g, i, j, k=1):
    g_params = [[g[i,i], g[i,j]], [g[j,i], g[j,j]]]
    
    x_min, x_max = float('inf'), -float('inf')
    min_coords, max_coords = None, None
    for theta in np.arange(0, 2*np.pi, 0.001):
        X = np.asarray([np.sin(theta), np.cos(theta)])
        epsilon = k / np.sqrt(np.dot(np.dot(X, g_params), X.T))
        x = epsilon*np.sin(theta)
        if x <= x_min:
            y = epsilon*np.cos(theta)
            min_coords = np.array((x,y))
            x_min = x
        if x >= x_max:
            x = epsilon*np.cos(theta)
            max_coords = np.array((x,y))
            x_max = x
    
    return np.linalg.norm(max_coords-min_coords)

def plot_confidences(confidence_gains, contrasts):
    save_path = "./results/confidence_gain"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    labels = [param.name for param in confidence_gains]
    for i, param in enumerate(confidence_gains):
        fig = plt.figure(figsize=[9,7], dpi=600)
        ax  = fig.add_subplot(111)
        
        gain = np.array(confidence_gains[param])
        for j in range(gain.shape[1]):
            if i != j:
                ax.plot(contrasts, gain[:,j], label=labels[j])
    
        ax.set_title(param.name)
        ax.set_xlabel("$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$", fontsize=11, weight='bold')
        ax.set_ylabel("Confidence Gain", fontsize=11, weight='bold')
        ax.legend()
        
        fig.savefig(save_path+"/DMPC_{}.png".format(i+1))

if __name__ == "__main__":
   angle_times = {0.7: (70, 50), #Angle: (Points, Time)
                  2.0: (70, 200)}
   initial = 6.36
   contrasts = np.arange(-0.56, 6.36, 0.2)
   #confidence_gain(initial, contrasts, angle_times)
   contrast_choice( contrasts, angle_times)