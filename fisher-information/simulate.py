import matplotlib.pyplot as plt
import numpy as np
import os

from typing import List, Tuple, Dict, Optional
from numpy.typing import ArrayLike

from refnx.dataset import ReflectDataset
from refnx.reflect import Structure, ReflectModel
from refnx.analysis import Objective

from plotting import plot_objective

DIRECTBEAM_PATH = '../fisher-information/data/directbeam/directbeam_wavelength.dat'
AngleTimes = Dict[float, Tuple[int, float]]

def simulate_single_contrast(structure: Structure, angle_times: AngleTimes, scale: float=1,
                             bkg: float=1e-7, dq: float=2, include_counts: bool=False,
                             save_path: str=None, file_name: str=None) -> Tuple[ReflectModel,
                                                                                ReflectDataset,
                                                                                Optional[ArrayLike]]:
    """Simulates a single contrast experiment measured over a number of angles.

    Args:
        structure (refnx.reflect.Structure): structure to simulate data for.
        angle_times (dict): points and simulation times for each angle.
        scale (float): experimental scale factor.
        bkg (float): level of the background to add.
        dq (float): instrument resolution.
        include_counts (bool): whether to return neutron counts or not.
        save_path (str): path to directory to save reflectivity data to.
        file_name (str): file name to use when saving reflectivity data.

    Returns:
        model (refnx.reflect.ReflectModel): model for given structure.
        dataset (refnx.reflect.ReflectDataset): simulated reflectivity data.
        counts (np.ndarray, optional): neutron counts for each Q value.

    """
    # Define the model.
    model = ReflectModel(structure, scale=scale, bkg=bkg, dq=dq)

    q, r, dr, counts = [], [], [], []
    total_points = 0
    for angle in angle_times:
        # Simulate the experiment for the angle.
        points, time = angle_times[angle]
        total_points += points
        q_angle, r_angle, dr_angle, counts_angle = run_experiment(model, angle, points, time)

        # Save combined dataset to .dat file if requested.
        str_angle = str(angle).replace('.', '')
        if save_path:
            if file_name:
                name = '{0}_{1}_simulated.dat'.format(file_name, str_angle)
                file_path = os.path.join(save_path, name)
            else:
                name = '{0}_simulated.dat'.format(str_angle)
                file_path = os.path.join(save_path, name)

            # Save reflectivity data as a 3-column ASCII file: Q, R, dR
            save_data = np.zeros((points, 3))
            save_data[:,0] = q_angle
            save_data[:,1] = r_angle
            save_data[:,2] = dr_angle
            np.savetxt(file_path, save_data, delimiter=',')

        # Combine the data for the angle with the data from other angles.
        q.append(q_angle)
        r.append(r_angle)
        dr.append(dr_angle)
        counts.append(counts_angle)

    data = np.zeros((total_points, 4))
    data[:,0] = np.concatenate(q)
    data[:,1] = np.concatenate(r)
    data[:,2] = np.concatenate(dr)
    data[:,3] = np.concatenate(counts)

    data = data[(data != 0).all(1)] # Remove points of zero reflectivity.
    data = data[data[:,0].argsort()] # Sort by Q.

    # Create refnx dataset for Q, R and dR values.
    dataset = ReflectDataset([data[:,0], data[:,1], data[:,2]])

    # Return neutron counts if requested.
    if include_counts:
        return model, dataset, data[:,3]
    else:
        return model, dataset

def simulate_multiple_contrasts(structures: List[Structure], angle_times: AngleTimes,
                                scale: float=1, bkg: float=1e-7, dq: float=2,
                                include_counts: bool=False, save_path: str=None
                                ) -> Tuple[List[ReflectModel],
                                           List[ReflectDataset],
                                           Optional[List[ArrayLike]]]:
    """Simulates a multiple contrast experiment measured over a number of angles.

    Args:
        structures (list): structures corresponding to each contrast.
        angle_times (dict): points and simulation times for each angle.
        scale (float): experimental scale factor.
        bkg (float): level of the background to add.
        dq (float): instrument resolution.
        include_counts (bool): whether to return neutron count or not.
        save_path (str): path to directory to save reflectivity data to.

    Returns:
        models (list): models for each contrast.
        datasets (list): simulated reflectivity data for each contrast.
        counts (list, optional): neutron counts corresponding to each Q value.

    """
    # Iterate over each structure.
    models, datasets, counts = [], [], []
    for i, structure in enumerate(structures, 1):
        # Simulate each measurement angle for the contrast.
        simulated = simulate_single_contrast(structure, angle_times,
                                             scale, bkg, dq, include_counts,
                                             save_path, 'contrast{}.dat'.format(i))
        models.append(simulated[0])
        datasets.append(simulated[1])
        counts.append(simulated[2])

    # Return neutron counts if requested.
    if include_counts:
        return models, datasets, counts
    else:
        return models, datasets

def run_experiment(model: ReflectModel, angle: float, points: int,
                   time: float, q_bin_edges: ArrayLike=None
                   ) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Simulates an experiment for a given model with added noise.

    Args:
        model (refnx.reflect.ReflectModel): model to simulate experiment on.
        angle (float): angle for simulation.
        points (int): number of points to use when binning.
        time (float): time to count during simulation.
        q_bin_edges (numpy.ndarray): edges of Q bins to use for simulation.

    Returns:
        q_binned (numpy.ndarray): Q values in equally log-spaced bins.
        r_noisy (numpy.ndarray): noisy reflectivity for each Q bin.
        r_error (numpy.ndarray): uncertainty in each reflectivity value.
        counts_incident (numpy.ndarray): incident neutrons for each Q bin.

    """
    # Load the directbeam_wavelength.dat file.
    direct_beam = np.loadtxt(DIRECTBEAM_PATH, delimiter=',')
    wavelengths = direct_beam[:,0] # 1st column is wavelength, 2nd is flux.

    # Adjust flux by measurement angle.
    direct_flux = direct_beam[:,1] * pow(angle/0.3, 2)

    theta = angle*np.pi / 180 # Convert angle from degrees to radians.
    q = 4*np.pi*np.sin(theta) / wavelengths # Calculate Q values.

    # If Q bins are not provided, bin Q values in equally log-spaced bins using flux as weighting.
    if q_bin_edges is None:
        q_bin_edges = np.logspace(np.log10(np.min(q)), np.log10(np.max(q)), points+1)

    flux_binned, _ = np.histogram(q, q_bin_edges, weights=direct_flux)

    # Get the bin centres and calculate model reflectivity.
    q_binned = np.asarray([(q_bin_edges[i] + q_bin_edges[i+1]) / 2 for i in range(points)])
    r_model = model(q_binned)

    # Calculate the number of incident neutrons for each bin.
    counts_incident = flux_binned*time

    # Get the measured reflected count for each bin (r_model accounts for background).
    counts_reflected = np.random.poisson(r_model*counts_incident).astype(float)

    # Convert from count space to reflectivity space.
    # Point has zero reflectivity if there is no flux.
    r_noisy = np.divide(counts_reflected, counts_incident,
                        out=np.zeros_like(counts_reflected), where=counts_incident!=0)

    r_error = np.divide(np.sqrt(counts_reflected), counts_incident,
                        out=np.zeros_like(counts_reflected), where=counts_incident!=0)

    return q_binned, r_noisy, r_error, counts_incident

def difference_plots(structure: Structure, angle_times: AngleTimes) -> None:
    """Plots the fractional difference in ground truth reflectivity and
       simulated reflectivity for a given structure.

    Args:
        structure (refnx.reflect.Structure): structure to simulate on.
        angle_times (dict): points and simulation times for each angle.

    """
    # Simulate an experiment using the given structure.
    objective = Objective(*simulate_single_contrast(structure(), angle_times))

    # Plot reflectivity curve.
    fig1, ax1 = plot_objective(objective)

    # Get the simulated and model reflectivity.
    data_sim = objective.data.y
    data_true = objective.model(objective.data.x)

    # Calculate the fractional error between the simulated and model data.
    true_sim_sum = np.asarray([sum(x) for x in zip(data_true, data_sim)])
    diff = (data_true - data_sim) / true_sim_sum

    # Error bars on difference.
    diff_error = 2*(objective.data.y_err / data_sim)*diff

    # Plot the difference against Q.
    fig2 = plt.figure(figsize=[9,7], dpi=600)
    ax2 = fig2.add_subplot(111)
    ax2.errorbar(objective.data.x, diff, diff_error, color="black", elinewidth=1, capsize=1.5)

    ax2.set_xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
    ax2.set_ylabel('Fractional Difference (arb.)', fontsize=11, weight='bold')
    ax2.set_xlim(0, 0.3)

if __name__ == '__main__':
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1, thin_layer_sample_2
    from structures import easy_sample, QCS_sample, many_param_sample

    structure = easy_sample
    angle_times = {0.7: (70, 5),
                   2.0: (70, 20)}

    # Display the difference plot between the model and simulated reflectivity.
    difference_plots(structure, angle_times)
