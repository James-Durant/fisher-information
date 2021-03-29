import matplotlib.pyplot as plt
import numpy as np
import os

from simulate import simulate

def difference_plots(structure, angle_times):
    save_path = "./results/difference_plots/"+structure.__name__
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for i in range(5):
        model, data = simulate(structure(), angle_times)

        data_sim = data.y
        data_true = model(data.x)
        
        fig1 = plt.figure(figsize=[9,7], dpi=600)
        ax1 = fig1.add_subplot(111)
        ax1.errorbar(data.x, data.y, data.y_err, color="black", marker="o", ms=3, lw=0, elinewidth=1, capsize=1.5)
        ax1.plot(data.x, data_true, color="red", zorder=20)
        ax1.set_xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
        ax1.set_ylabel('Reflectivity (arb.)', fontsize=11, weight='bold')
        ax1.set_yscale('log')
        ax1.set_xlim(0.005, 0.25)
        ax1.set_ylim(1e-7, 2)

        diff = (data_true - data_sim) / np.asarray([sum(x) for x in zip(data_true, data_sim)]) 
        diff_error = 2*(data.y_err / data.y)*diff
        
        fig2 = plt.figure(figsize=[9,7], dpi=600)
        ax2 = fig2.add_subplot(111)
        ax2.errorbar(data.x, diff, diff_error, color="black", elinewidth=1, capsize=1.5)
        ax2.set_xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
        ax2.set_ylabel('Difference (arb.)', fontsize=11, weight='bold')
        ax2.set_xlim(0.005, 0.25)
        
        fig1.savefig(save_path+'/{}_reflectivity.png'.format(i+1))
        fig2.savefig(save_path+'/{}_difference.png'.format(i+1))
        
if __name__ == "__main__":
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1, thin_layer_sample_2
    from structures import easy_sample, many_param_sample

    structure = easy_sample
    angle_times = {0.7: (70, 5), #Angle: (Points, Time)
                   2.0: (70, 20)}

    difference_plots(structure, angle_times)