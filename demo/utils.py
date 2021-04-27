import matplotlib.pyplot as plt
import numpy as np
import os

from typing import List, Dict, Tuple, Optional, Callable
from numpy.typing import ArrayLike

from refnx.dataset import ReflectDataset
from refnx.reflect import Structure, ReflectModel
from refnx.analysis import Parameter, Objective

DIRECTBEAM_PATH = './directbeam/directbeam_wavelength.dat'
AngleTimes = Dict[float, Tuple[int, float]]

def simulate_single_contrast(structure: Structure, angle_times: AngleTimes,
                             scale: float=1, bkg: float=1e-7, dq: float=2,
                             include_counts: bool=False, save_path: str=None,
                             file_name: str=None
                             ) -> Tuple[ReflectModel,
                                        ReflectDataset,
                                        Optional[ArrayLike]]:
    """Simulates a single contrast experiment measured over a number of angles.

    Args:
        structure (refnx.reflect.Structure): structure to simulate on.
        angle_times (dict): points and times for each measured angle.
        scale (float): instrument experimental scale factor.
        bkg (float): level of the background to add.
        dq (float): instrument resolution.
        include_counts (bool): whether to return neutron counts or not.
        save_path (str): path to directory to save reflectivity data to.
        file_name (str): file name to use when saving reflectivity data.

    Returns:
        model (refnx.reflect.ReflectModel): model for the given structure.
        dataset (refnx.reflect.ReflectDataset): simulated reflectivity data.
        counts (np.ndarray, optional): neutron counts for each Q value.

    """
    # Define the model.
    model = ReflectModel(structure, scale=scale, bkg=bkg, dq=dq)

    q, r, r_error, counts = [], [], [], []
    total_points = 0
    for angle in angle_times:
        # Simulate the experiment for the angle.
        points, time = angle_times[angle]
        total_points += points
        simulated = run_experiment(model, angle, points, time)
        q_angle, r_angle, r_error_angle, counts_angle = simulated

        # Save combined dataset to .dat file if requested to save data.
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
            save_data[:,2] = r_error_angle
            np.savetxt(file_path, save_data, delimiter=',')

        # Combine the data for the angle with the data from other angles.
        q += q_angle
        r += r_angle
        r_error += r_error_angle
        counts += counts_angle

    data = np.zeros((total_points, 4))
    data[:,0] = q
    data[:,1] = r
    data[:,2] = r_error
    data[:,3] = counts

    data = data[(data != 0).all(1)] # Remove points of zero reflectivity.
    data = data[data[:,0].argsort()] # Sort by Q.

    # Create a refnx dataset for Q, R and dR values.
    dataset = ReflectDataset([data[:,0], data[:,1], data[:,2]])

    # Return the associated neutron counts if requested.
    if include_counts:
        return model, dataset, data[:,3]
    else:
        return model, dataset

def run_experiment(model: ReflectModel, angle: float, points: int, time: float,
                   q_bin_edges: ArrayLike=None
                   ) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Simulates an experiment for a given model with added noise.

    Args:
        model (refnx.reflect.ReflectModel): model to simulate experiment on.
        angle (float): measurement angle for simulation.
        points (int): number of points to use when binning.
        time (float): time to count during experiment.
        q_bin_edges (np.ndarray): edges of the Q bins to use for simulation.

    Returns:
        q_binned (list): Q values in equally log-spaced bins.
        r (list): noisy reflectivity for each Q bin.
        r_errors (list): errors in each reflectivity value.
        counts (list): total number of incident neutrons for each Q bin.

    """
    # Load the directbeam_wavelength.dat file.
    direct_beam = np.loadtxt(DIRECTBEAM_PATH, delimiter=',')
    wavelengths = direct_beam[:,0] # 1st column is wavelength, 2nd is flux.

    # Adjust flux by measurement angle.
    direct_flux = direct_beam[:,1] * pow(angle/0.3, 2)

    theta = angle*np.pi / 180 # Convert angle from degrees into radians.
    q = 4*np.pi*np.sin(theta) / wavelengths # Calculate Q values.

    # If Q bins are not provided
    if q_bin_edges is None:
        # Bin Q values in equally log-spaced bins using flux as weighting.
        q_bin_edges = np.logspace(np.log10(np.min(q)),
                                  np.log10(np.max(q)), points+1)

    flux_binned, _ = np.histogram(q, q_bin_edges, weights=direct_flux)

    # Get the bin centres and calculate model reflectivity.
    q_binned = [(q_bin_edges[i] + q_bin_edges[i+1]) / 2 for i in range(points)]
    reflectance = model(q_binned)

    r, r_errors, counts = [], [], []
    for i in range(points): # Iterate over the desired number of points (bins).
        flux_point = flux_binned[i]
        r_point    = reflectance[i]
        count_incident = flux_point * time

        # Get the measured reflected count for the bin.
        # r_point accounts for background.
        count_reflected = np.random.poisson(r_point*flux_point*time)

        # Point has zero reflectivity if there is no flux.
        if count_reflected > 0:
            # Convert from count space to reflectivity space.
            r_noisy = count_reflected / count_incident
            r_error = np.sqrt(count_reflected) / count_incident

            r.append(r_noisy)
            r_errors.append(r_error)
            counts.append(count_incident) # Incident neutrons in the bin.
        else:
            r.append(0)
            r_errors.append(0)
            counts.append(0)

    return q_binned, r, r_errors, counts

def first_angle_choice(structure: Callable, angles: ArrayLike, points: int,
                       time: float, save_path: str) -> None:
    """Investigates how the FIM changes, for each parameter of a given
       `structure`, with first angle choice.

    Args:
        structure (function): structure to calculate the FIM with.
        angles (numpy.ndarray): measurement angles to calculate FIM with.
        points (int): number of points to simulate for each angle.
        time (float): measurement time to use when simulating each angle.
        save_path (str): path to directory to save FIM plot to.

    """
    save_path = os.path.join(save_path, structure.__name__)

    # Iterate over each angle in the given array.
    information = []
    for i, angle in enumerate(angles):
        angle_times = {angle: (points, time)}

        # Simulate data for the given structure with current angle.
        simulated = simulate_single_contrast(vary_structure(structure()),
                                             angle_times, include_counts=True)
        model, data, counts = simulated

        objective = Objective(model, data)
        xi = objective.varying_parameters()

        # Calculate FIM matrix and individual parameter FIMs.
        g = fisher_single_contrast(data.x, xi, counts, model)
        information.append(np.diag(g))

        # Display progress.
        if i % 10 == 0:
            print('>>> {0}/{1}'.format(i, len(angles)))
        
    print()
    plot_information(angles, information, xi, save_path, 'first_angle')

def second_angle_choice(structure: Callable, initial_angle_times: AngleTimes,
                        angles: ArrayLike, points: int, time: float,
                        save_path: str) -> None:
    """Investigates how the FIM changes, for each parameter of a given
       `structure`, with second angle choice assuming an `initial_angle`
       was first measured.

    Args:
        structure (function): structure to calculate the FIM with.
        initial_angle_times (dict): angles initially measured.
        angles (numpy.ndarray): measurement angles to calculate FIM with.
        points (int): number of points to simulate for each angle.
        time (float): measurement time to use when simulating each angle.
        save_path (str): path to directory to save FIM plot to.

    """
    save_path = os.path.join(save_path, structure.__name__)

    sample = vary_structure(structure())

    simulated_initial = simulate_single_contrast(sample, initial_angle_times,
                                                 include_counts=True)
    model_initial, data_initial, counts_initial = simulated_initial

    objective = Objective(model_initial, data_initial)
    xi = objective.varying_parameters()

    # Iterate over each angle in the given array.
    information = []
    for i, angle in enumerate(angles):
        new_angle_times = {angle: (points, time)}

        simulated_new = simulate_single_contrast(sample, new_angle_times,
                                                 include_counts=True)
        model_new, data_new, counts_new = simulated_new

        qs = [data_initial.x, data_new.x]
        counts = [counts_initial, counts_new]
        models = [model_initial, model_new]

        # Calculate the FIM matrix for the two contrast dataset.
        g = fisher_multiple_contrasts(qs, xi, counts, models)
        information.append(np.diag(g))

        # Display progress.
        if i % 10 == 0:
            print('>>> {0}/{1}'.format(i, len(angles)))

    print()
    plot_information(angles, information, xi, save_path, 'second_angle')

def first_contrast_choice(bilayer, contrasts: ArrayLike,
                          angle_times: AngleTimes, save_path: str) -> None:
    """Investigates how the FIM changes, for each parameter of a `bilayer`
       model, for each contrast SLD in a given array of `contrasts`.

    Args:
        bilayer (Bilayer): bilayer to find optimal contrast for.
        contrasts (numpy.ndarray): contrast SLDs to calculate FIM with.
        angle_times (dict): points and measurement times for each angle.
        save_path (str): path to directory to save FIM plot to.

    """
    save_path = os.path.join(save_path, bilayer.name)
    xi = bilayer.parameters

    # Iterate over each SLD in the given array.
    information = []
    for i, contrast_sld in enumerate(contrasts):
        # Simulate data for the given bilayer model with current contrast SLD.
        structure = bilayer.using_contrast(contrast_sld)
        model, data, counts = simulate_single_contrast(structure, angle_times,
                                                       include_counts=True)
        # Calculate FIM matrix and record the diagonal elements.
        g = fisher_single_contrast(data.x, xi, counts, model)
        information.append(np.diag(g))

        # Display progress.
        if i % 10 == 0:
            print('>>> {0}/{1}'.format(i, len(contrasts)))
            
    print()
    # Plot the FIM as a function of contrast SLD.
    plot_information(contrasts, information, xi, save_path, 'first_contrast')

def second_contrast_choice(bilayer, initial_contrast: float,
                           contrasts: ArrayLike, angle_times: AngleTimes,
                           save_path: str) -> None:
    """Investigates how the FIM changes, for each parameter of a `bilayer`
       model, for each second measurement contrast SLD in a given array
       of `contrasts`.

    Args:
        bilayer (Bilayer): bilayer to find optimal contrast for.
        initial_contrast (float): initial measured contrast SLD.
        contrasts (numpy.ndarray): second contrast SLDs to calculate FIM with.
        angle_times (dict): points and measurement times for each angle.
        save_path (str): path to directory to save FIM plot to.

    """
    save_path = os.path.join(save_path, bilayer.name)
    xi = bilayer.parameters

    # Simulate an experiment using the given initial contrast.
    structure = bilayer.using_contrast(initial_contrast)
    simulated_initial = simulate_single_contrast(structure, angle_times,
                                                 include_counts=True)
    model_initial, data_initial, counts_initial = simulated_initial

    # Iterate over each second contrast SLD in the given array.
    information = []
    for i, contrast_sld in enumerate(contrasts):
        # Simulate data for the given bilayer model with current contrast SLD.
        structure = bilayer.using_contrast(contrast_sld)

        simulated_new = simulate_single_contrast(structure, angle_times,
                                                 include_counts=True)
        model_new, data_new, counts_new = simulated_new

        # Combine the data from the initial and second contrasts.
        qs = [data_initial.x, data_new.x]
        counts = [counts_initial, counts_new]
        models = [model_initial, model_new]

        # Calculate the FIM matrix for the two contrast dataset.
        g = fisher_multiple_contrasts(qs, xi, counts, models)
        information.append(np.diag(g))

        # Display progress.
        if i % 10 == 0:
            print('>>> {0}/{1}'.format(i, len(contrasts)))
            
    print()
    # Plot the FIM as a function of second contrast SLD.
    plot_information(contrasts, information, xi, save_path, 'second_contrast')

def thickness_choice(bilayer, thicknesses, contrast_sld, angle_times, save_path):
    """Investigates how the FIM changes, for each parameter of a bilayer
       model, with underlayer (SiO2) thickness.

    Args:
        bilayer (Bilayer): bilayer model to find optimal thickness for.
        thicknesses (numpy.ndarray): thicknesses to calculate FIM over.
        contrast_sld (float): SLD of contrast.
        angle_times (dict): points and measurement times for each angle.
        save_path (str): path to directory to save FIM plot to.

    """
    save_path = os.path.join(save_path, bilayer.name)
    xi = []
    for param in bilayer.parameters:
        if param.name == 'SiO2 Thickness':
            sio2_thick = param
        else:
            xi.append(param)

    # Iterate over each thickness in the given array.
    information = []
    for i, thickness in enumerate(thicknesses):
        # Simulate data for the given bilayer model with current SiO2 thickness.
        sio2_thick.value = thickness
        
        structure = bilayer.using_contrast(contrast_sld)
        model, data, counts = simulate_single_contrast(structure, angle_times,
                                                       include_counts=True)
        # Calculate the FIM
        g = fisher_single_contrast(data.x, xi, counts, model)
        information.append(np.diag(g))

        # Display progress.
        if i % 10 == 0:
            print('>>> {0}/{1}'.format(i, len(thicknesses)))
    
    print()
    # Plot the FIM as a function of SiO2 thickness.
    plot_information(thicknesses, information, xi, save_path, 'thickness')

def plot_information(x: ArrayLike, information: ArrayLike,
                     xi: List[Parameter], save_path: str, x_label: str) -> None:
    """Plots the FIM for each parameter of a model against given `x` array.

    Args:
        x (numpy.ndarray): values to plot FIM values against.
        information (numpy.ndarray): FIM values for each parameter.
        xi (list): model parameters.
        save_path (str): path to directory to save FIM plot to.
        x_lable (str): either 'contrast', 'angle' or 'thickness'.
        normalise (bool): whether to normalise the FIM to [0,1].

    """
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)

    # Plot the FIM values for each free parameter.
    information = np.asarray(information)
    for i, param in enumerate(xi):
        ax.plot(x, information[:,i], label=param.name)

    ax.legend() # Create legend.

    # Add the correct x-axis label.
    if x_label == 'first_contrast' or x_label == 'second_contrast':
        ax.set_xlabel('$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$',
                      fontsize=11, weight='bold')
    elif x_label == 'angle':
        ax.set_xlabel('Angle (°)', fontsize=11, weight='bold')
    elif x_label == 'thickness':
        ax.set_xlabel('$\mathregular{Thickness\ (\AA)}$', fontsize=11, weight='bold')

    # Save both linear and log scale plots.
    ax.set_ylabel('Fisher Information (arb.)', fontsize=11, weight='bold')
    save_plot(fig, save_path, '{}_choice_linear'.format(x_label))

    ax.set_yscale('log')
    ax.legend(loc='upper right')
    save_plot(fig, save_path, '{}_choice_log'.format(x_label))

def save_plot(fig: plt.Figure, save_path: str, file_name: str) -> None:
    """Saves a figure to a given directory.

    Args:
        fig (matplotlib.pyplot.Figure): figure to save.
        save_path (str): path to directory to save the figure to.
        file_name (str): name of file to save the plot as.

    """
    # Create the directory if not present.
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = os.path.join(save_path, file_name+'.png')
    fig.savefig(file_path, dpi=600)

def vary_structure(structure: Structure, random_init: bool=False,
                   bound_size: float=0.2, vary_sld: bool=True,
                   vary_thick: bool=True, vary_rough: bool=False) -> Structure:
    """Vary the parameters of each layer of a given `structure` and optionally,
       initialise these values to random values within their bounds.

    Args:
        structure (refnx.reflect.Structure): structure to vary.
        random_init (bool): whether to randomly initialise parameters.
        bound_size (float): size of the bounds to place on the parameters.
        vary_sld (bool): whether to vary the structure's layers' SLDs.
        vary_thick (bool): whether to vary structure's layers' thicknesses.
        vary_rough (bool): whether to vary structure's layers' roughnesses.

    Returns:
        refnx.reflect.Structure: a reference to the given structure.

    """
    # Skip over air and the substrate.
    for component in structure.components[1:-1]:
        # Vary each layers' SLD, thickness and roughness if requested.
        if vary_sld:
            sld_bounds = (component.sld.real.value*(1-bound_size),
                          component.sld.real.value*(1+bound_size))
            component.sld.real.setp(vary=True, bounds=sld_bounds)
            # Set parameter to an arbitrary initial value within its bounds.
            if random_init:
                component.sld.real.value = np.random.uniform(*sld_bounds)

        if vary_thick:
            thick_bounds = (component.thick.value*(1-bound_size),
                            component.thick.value*(1+bound_size))
            component.thick.setp(vary=True, bounds=thick_bounds)
            if random_init:
                component.thick.value = np.random.uniform(*thick_bounds)

        if vary_rough:
            rough_bounds = (component.rough.value*(1-bound_size),
                            component.rough.value*(1+bound_size))
            component.rough.setp(vary=True, bounds=rough_bounds)
            if random_init:
                component.rough.value = np.random.uniform(*rough_bounds)

    # Vary the substrate's roughness.
    if vary_rough:
        component = structure.components[-1]
        rough_bounds = (component.rough.value*(1-bound_size),
                        component.rough.value*(1+bound_size))
        component.rough.setp(vary=True, bounds=rough_bounds)
        if random_init:
            component.rough.value = np.random.uniform(*rough_bounds)

    return structure

def fisher_single_contrast(q: ArrayLike, xi: List[Parameter], counts: ArrayLike,
                           model: ReflectModel) -> ArrayLike:
    """Calculates the FIM matrix for a given `model`.

    Args:
        q (numpy.ndarray): array of Q values.
        xi (list): parameters representing each varying parameter.
        counts (numpy.ndarray): incident neutron counts for each Q value.
        model (refnx.reflect.ReflectModel): model for calculating gradients.

    Returns:
        numpy.ndarray: FIM matrix for the model and data.

    """
    n = len(q)
    m = len(xi)
    J = np.zeros((n,m))
    # Calculate the gradient of the model reflectivity with every model
    # parameter for every model data point.
    for i in range(n):
        for j in range(m):
            J[i,j] = gradient(model, xi[j], q[i])

    r = model(q) #Use model reflectivity values
    M = np.diag(counts/r, k=0)
    return np.dot(np.dot(J.T, M), J)

def fisher_multiple_contrasts(qs: List[ArrayLike], xi: List[Parameter],
                              counts: List[ArrayLike],
                              models: List[ReflectModel]) -> ArrayLike:
    """Calculates the Fisher information metric (FIM) matrix for a given list
       of `models` and set of parameters, `xi`.

    Args:
        qs (list): Q arrays corresponding to each contrast.
        xi (list): varying parameters.
        counts (list): incident neutron counts corresponding to each Q value.
        models (list): models to calculate gradients with.

    Returns:
        numpy.ndarray: FIM matrix for the given models and parameters.

    """
    n = sum(len(q) for q in qs) # Number of data points.
    m = len(xi) # Number of parameters.
    J = np.zeros((n,m))

    # Calculate the gradient of the model reflectivity with every model
    # parameter for every model data point.
    r_all = []
    start = 0
    for q, model in list(zip(qs, models)):
        for i in range(len(q)):
            for j in range(m):
                J[start+i,j] = gradient(model, xi[j], q[i])

        start += len(q)
        r_all.append(model(q)) # Use model reflectivity values.

    r = np.concatenate(r_all)
    M = np.diag(np.concatenate(counts) / r, k=0)
    return np.dot(np.dot(J.T, M), J)

def gradient(model: ReflectModel, parameter: Parameter, q_point: float,
             step: float=0.005) -> float:
    """Calculate two-point gradient of model reflectivity with model parameter.

    Args:
        model (refnx.reflect.ReflectModel): model to calculate gradient.
        parameter (refnx.analysis.Parameter): parameter to vary.
        q_point (float): Q value of the R point to calculate the gradient of.
        step (float): step size to take when calculating the gradient.

    Returns:
        float: two-point gradient.

    """
    old, step = parameter.value, parameter.value*step # 0.5% step by default.

    x1 = parameter.value = old - step # First point
    y1 = model(q_point) # Get new r value with altered model.

    x2 = parameter.value = old + step # Second point
    y2 = model(q_point)

    parameter.value = old # Reset parameter
    return (y2-y1) / (x2-x1) # Return the gradient
