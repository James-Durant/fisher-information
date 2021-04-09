import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append('./') # MCMC sampling cannot find structures code without this?

from typing import List, Tuple, Dict
from numpy.typing import ArrayLike

from scipy.stats import t

from refnx.dataset import ReflectDataset
from refnx.reflect import ReflectModel
from refnx.analysis import Objective

from structures import QCS_sample
from simulate import run_experiment

from plotting import save_plot, plot_objective
from utils import vary_structure, Sampler

RefData = Dict[float, Tuple[ArrayLike, ArrayLike, ArrayLike]]

def sample_measured_data(data_path: str, bkg: float, dq: float,
                         save_path: str) -> None:
    """Samples data in the 'QCS_all.dat' file using MCMC and nested sampling.

    Args:
        data_path (str): path to directory containing 'QCS_all.dat' file.
        bkg (float): model background.
        dq (float): model instrument resolution.
        save_path (str): path to directory to save sampling corner plots to.

    """
    # Create the QCS sample and vary the SLDs, thicknesses and roughnesses.
    model = ReflectModel(vary_structure(QCS_sample(), vary_rough=True),
                         scale=1, bkg=bkg, dq=dq)

    # Load and scale the data so that the maximum reflectivity is 1.
    data = ReflectDataset(data_path)
    data.scale(np.max(data.y))

    # Sample using MCMC and nested sampling and save the corner plots.
    sampler = Sampler(Objective(model, data))
    fig1 = sampler.sample_MCMC(verbose=True)
    save_plot(fig1, save_path, 'measured_sample_MCMC')
    fig2 = sampler.sample_nested(verbose=True, dynamic=True)
    save_plot(fig2, save_path, 'measured_sample_nested.png')

def plot_measured_all(data_path: str, bkg: float, dq: float,
                      save_path: str) -> None:
    """Plots the binned measured data (combined data for all angles).

    Args:
        data_path (str): path to directory containing 'QCS_all.dat' file.
        bkg (float): model background.
        dq (float): model instrument resolution.
        save_path (str): path to directory to save plot to.

    """
    # Define the model, load and scale the data.
    model = ReflectModel(QCS_sample(), scale=1, bkg=bkg, dq=dq)
    data = ReflectDataset(data_path)
    # Scale the data so that the maximum reflectivity is 1.
    data.scale(np.max(data.y))

    # Plot and save the fit of the measured dataset.
    fig, _ = plot_objective(Objective(model, data))
    save_plot(fig, save_path, 'measured_fit')

def simulate_measured_data(data_path: str, files: List[str], scale: float,
                           bkg: float, dq: float, angles: List[float],
                           time: float) -> Tuple[RefData, RefData]:
    """Loads each measured angle's data and simulates experiments for each.

    Args:
        data_path (str): path to directory containing the measured data.
        files (list): list of file names for each angle's data.
        scale (float): instrument experimental scale factor.
        bkg (float): instrument background parameter.
        dq (float): instrument resolution.
        angles (list): angles used in measuring the data.
        time (float): measurement time for the measured data.

    Returns:
        measured (dict): measured reflectivity data for each angle.
        simulated (dict): simulated reflectivity data for each angle.

    """
    # Iterate over each measurement angle and corresponding file.
    measured, simulated = {}, {}
    for angle, file_name in list(zip(angles, files)):
        # Load the measured reflectivity data.
        data_measured = np.loadtxt(os.path.join(data_path, file_name),
                                   delimiter='    ')
        q_measured = data_measured[:,0]
        r_measured = data_measured[:,1]
        r_error_measured = data_measured[:,2]
        measured[angle] = (q_measured, r_measured, r_error_measured)

        # For simulation, we need the edges of the Q bins but the measured data
        # contains Q bin centres so approximate the edges.
        points = len(q_measured)
        temp = [q_measured[0]-(q_measured[1]-q_measured[0])/2]
        temp += q_measured.tolist()
        q_bins = [(temp[i] + temp[i+1]) / 2 for i in range(points)]
        q_bins += [q_measured[-1]+(q_measured[-1]-q_measured[-2])/2]

        # Define the model with correct scale and simulate.
        model = ReflectModel(QCS_sample(), scale=scale, bkg=bkg, dq=dq)
        data_simulated = run_experiment(model, angle, points, time, q_bins)
        q_simulated = data_simulated[0]
        r_simulated = data_simulated[1]
        r_error_simulated = data_simulated[2]
        simulated[angle] = (q_simulated, r_simulated, r_error_simulated)

    return measured, simulated

def plot_angle_data(angle_data: RefData, data_type: str,
                    save_path: str) -> None:
    """Creates a plot overlaying each dataset of each measured/simulated angle.

    Args:
        angle_data (dict): dictionary of reflectivity data for each angle.
        data_type (str): either 'Measured' or 'Simulated'.
        save_path (str): path to directory to save plot to.

    """
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)

    #Iterate over each simulated/measured angle.
    for angle in angle_data:
        q, r, r_error = angle_data[angle]
        # Plot Q values vs. reflectivity with associated reflectivity error bars.
        ax.errorbar(q, r, r_error, marker='o', ms=3, lw=0, elinewidth=1,
                    capsize=1.5, label='{0} {1}°'.format(data_type, str(angle)))

    ax.set_xlabel('$\mathregular{Q\ (Å^{-1})}$', fontsize=11, weight='bold')
    ax.set_ylabel('Reflectivity (arb.)', fontsize=11, weight='bold')
    ax.set_yscale('log')
    ax.set_xlim(0, 0.18) # Only plot up to the maximum Q in the measured data.
    ax.set_ylim(1e-7, 2)
    ax.legend()

    save_plot(fig, save_path, '{}_reflectivity.png'.format(data_type.lower()))

def similarity(measured: RefData, simulated: RefData) -> Tuple[float, float]:
    """Determines whether there is significant difference between given
       measured and simulated reflectivity datasets.

    Args:
        measured (dict): measured reflectivity data for each angle.
        simulated (dict): simulated reflectivity data for each angle.

    Returns:
        t_statistic (float): Hotelling T-squared statistic.
        pval (float): associated p-value for t-statistic.

    """
    # Get the measured and simulated reflectivity and errors for all angles.
    # The data is simulated at the same Q values as the measured data.
    r_measured = np.asarray([measured[angle][1]
                             for angle in measured]).flatten()
    
    r_simulated = np.asarray([simulated[angle][1]
                              for angle in simulated]).flatten()
    
    r_error_measured = np.asarray([measured[angle][2]
                                   for angle in measured]).flatten()
    
    r_error_simulated = np.asarray([simulated[angle][2]
                                    for angle in simulated]).flatten()

    # Calculate the measured and reflected counts, taking the count as 0
    # if the reflectivity error is 0.
    counts_measured = np.divide(r_measured, r_error_measured,
                                out=np.zeros_like(r_measured),
                                where=r_error_measured!=0)

    counts_simulated = np.divide(r_simulated, r_error_simulated,
                                 out=np.zeros_like(r_simulated),
                                 where=r_error_simulated!=0)

    # Apply an Anscombe transformation to make the values approximately normal.
    # Then weight each value by 1 / standard deviation
    counts_measured = 2*np.sqrt(counts_measured + 3/8) / np.std(counts_measured)
    counts_simulated = 2*np.sqrt(counts_simulated + 3/8) / np.std(counts_simulated)

    # Calculate the mean over the squared differences in reflectivity values.
    # Then take the square root of this value to get a Hotelling T-statistic.
    t_statistic = np.sqrt(np.mean(np.square(counts_measured-counts_simulated)))

    # Get the associated p-value using the SciPy survival function.
    pval = t.sf(np.abs(t_statistic), len(r_measured)-1)*2
    return t_statistic, pval

if __name__ == '__main__':
    save_path = './results/QCS_sample'
    data_path = './data/QCS_sample'

    scale = 0.783
    bkg = 8e-7
    dq = 2.5

    # Plot the binned QCS data and sample it.
    all_data_path = os.path.join(data_path, 'QCS_all.dat')
    sample_measured_data(all_data_path, bkg, dq, save_path)
    plot_measured_all(all_data_path, bkg, dq, save_path)

    angles = [0.3, 0.4, 0.5, 0.6, 0.7, 2.0]
    measured_data = ['QCS_03_1uA.dat', 'QCS_04_1uA.dat', 'QCS_05_1uA.dat',
                     'QCS_06_1uA.dat', 'QCS_07_1uA.dat', 'QCS_20_1uA.dat']

    # Plot the measured individual angle data and their simulated counterparts.
    measured, simulated = simulate_measured_data(data_path, measured_data,
                                                 scale, bkg, dq, angles,
                                                 time=1)
    plot_angle_data(measured, 'Measured', save_path)
    plot_angle_data(simulated, 'Simulated', save_path)

    # Compute the similarity between the measured and simulated data.
    t_statistic, pval = similarity(measured, simulated)
    with open(os.path.join(save_path, 'similarity.txt'), 'w') as file:
        file.write('Hotelling t-squared statistic: {}\n'.format(t_statistic))
        file.write('p-value: {}'.format(pval))