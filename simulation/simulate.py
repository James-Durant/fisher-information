import numpy as np
import matplotlib.pyplot as plt

from refnx.reflect import ReflectModel

def simulate_single_contrast(structure, angle_times, dq=2, bkg=1e-7, bkg_rate=1e-7, save_path=None, filename="data.dat"):
    """Simulates a single contrast experiment measured using a number of different angles.

    Args:
        structure (refnx.reflect.Structure): structure to simulate the experiment on.
        angle_times (dict): dictionary of number of points and times for each measured angle.
        dq (float): value to use for the model's instrument resolution parameter.
        bkg (float): value to use for the model's background parameter.
        bkg_rate (float): the level of the background noise.
        save_path (string): path to directory to save the reflectivity data to.
        filename (string): file name to use when storing reflectivity data.

    Returns:
        model (refnx.reflect.ReflectModel): the model for the given structure.
        data (numpy.ndarray): reflectivity data and corresponding flux values.

    """
    model = ReflectModel(structure, scale=1, dq=dq, bkg=bkg)

    q, r, r_error, flux = [], [], [], []
    total_points = 0
    for angle in angle_times:
        #Simulate the experiment for the angle.
        points, time = angle_times[angle]
        q_angle, r_angle, r_error_angle, flux_angle = run_experiment(model, angle, points, time, bkg_rate)

        q += q_angle
        r += r_angle
        r_error += r_error_angle
        flux += flux_angle
        total_points += points

    data = np.zeros((total_points, 4))
    data[:,0] = q
    data[:,1] = r
    data[:,2] = r_error
    data[:,3] = flux

    data = data[(data != 0).all(1)]  #Remove points of zero reflectivity
    data = data[data[:,0].argsort()] #Sort by Q

    #Save combined dataset to .dat file
    if save_path:
        np.savetxt(save_path+"/"+filename, data[:,:-1], delimiter=",")

    return model, data

def simulate_multiple_contrasts(structures, angle_times, dq=2, bkg=1e-7, bkg_rate=1e-6, save_path=None,):
    """Simulates a multiple contrast experiment measured using a number of different angles.

    Args:
        structures (list): list of refnx.reflect.Structure objects corresponding to each contrast.
        angle_times (dict): dictionary of number of points and times for each measured angle.
        dq (float): value to use for the model's instrument resolution parameter.
        bkg (float): value to use for the model's background parameter.
        bkg_rate (float): the level of the background noise.
        save_path (string): path to directory to save the reflectivity data for each contrast to.

    Returns:
        models (list): model for each contrast.
        datasets (list): experimental data for each contrast.

    """
    models, datasets = [], []
    for i, structure in enumerate(structures, 1):
        filename = "contrast_{}.dat".format(i)
        #Simulate each measurement angle for the contrast.
        model, data = simulate_single_contrast(structure, angle_times, dq, bkg,
                                               bkg_rate, save_path, filename)
        models.append(model)
        datasets.append(data)

    return models, datasets

def run_experiment(model, angle, points, time, bkg_rate=1e-6, q_min=None, q_max=None, directbeam_file="../simulation/data/directbeam_wavelength.dat"):
    """Simulates an experiment for a given `model` with added noise.

    Args:
        model (refnx.reflect.ReflectModel): the model to simulate the experiment with.
        angle (float): the measurement angle for the simulation.
        points (int): the number of points to use when binning.
        time (int): the amount of time to count for during the experiment.
        bkg_rate (float): the level of the background noise.
        q_min (float): minimum Q value for range of simulation.
        q_max (float): maximum Q value for range of simulation.
        directbeam_file (string): the file path to the directbeam_wavelength.dat file.

    Returns:
        q_binned (list): Q values in equally log-spaced bins.
        r (list): noisy reflectivity for each Q value.
        r_errors (list): errors in each reflectivity value.
        counts (list): total number of incident neutrons for each Q value.

    """
    #Load the directbeam_wavelength.dat file.
    direct_beam  = np.loadtxt(directbeam_file, delimiter=',')
    wavelengths  = direct_beam[:,0] # 1st column is wavelength, 2nd is flux.
    #Adjust flux by measurement angle.
    direct_flux = direct_beam[:,1] * pow(angle/0.3, 2)

    theta = angle*np.pi / 180 #Convert angle from degrees into radians.
    q = 4*np.pi*np.sin(theta) / wavelengths #Calculate Q values.

    if q_min is None:
        q_min = np.min(q)
    if q_max is None:
        q_max = np.max(q)

    #Bin Q values in equally log-spaced bins using flux as weighting.
    log_q = np.logspace(np.log10(q_min), np.log10(q_max), points+1)
    flux_binned, q_bin_edges = np.histogram(q, log_q, weights=direct_flux)

    #Get the bin centres and calculate model reflectivity.
    q_binned = [(q_bin_edges[i] + q_bin_edges[i+1]) / 2 for i in range(points)]
    reflectance = model(q_binned)

    r, r_errors, counts = [], [], []
    for i in range(points): #Iterate over the desired number of points (bins).
        flux_point  = flux_binned[i]
        r_point     = reflectance[i]
        count_point = flux_point * time

        #Add background noise and get the measured reflected count for the bin.
        count_noisy = np.random.poisson((r_point*flux_point + bkg_rate) * time)
        
        #Point has zero reflectivity if there is no flux.
        if count_noisy > 0:
            #Convert from count space to reflectivity space.
            r_noisy = count_noisy / count_point
            r_error = np.sqrt(count_noisy) / count_point
            
            r.append(r_noisy)
            r_errors.append(r_error)
            counts.append(count_noisy / r_noisy) #Incident neutrons in the bin
        else:
            r.append(0)
            r_errors.append(0)
            counts.append(0) #Incident neutrons in the bin

    return q_binned, r, r_errors, counts

def vary_model(model):
    """Vary the SLD and thickness of each layer of a given `model` and initialise
       these values to random values within their bounds.

    Args:
        model (refnx.reflect.ReflectModel): the model to vary.

    """
    #Skip over Air/D20 and substrate.
    for component in model.structure.components[1:-1]:
        #Use a bound of 50% above and below the ground truth value.
        sld_bounds   = (component.sld.real.value*0.5, component.sld.real.value*1.5)
        thick_bounds = (component.thick.value*0.5,    component.thick.value*1.5)

        #Vary the SLD and thickness of the layer.
        component.sld.real.setp(vary=True, bounds=sld_bounds)
        component.thick.setp(vary=True, bounds=thick_bounds)

        #Set the SLD and thickness to arbitrary initial values (within their bounds).
        component.sld.real.value = np.random.uniform(*sld_bounds)
        component.thick.value    = np.random.uniform(*thick_bounds)

def plot_objective(objective, show_fit=True):
    """Plots the fit of a given `objective` against the objective's data.

    Args:
        objective (refnx.analysis.Objective): the objective to plot.
        show_fit (Boolean): whether to display the objective fit or not.

    """
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)

    #Add the data in a transformed fashion.
    y, y_err, model = objective._data_transform(model=objective.generative())
    ax.errorbar(objective.data.x, y, y_err, color="black",
                marker="o", ms=3, lw=0, elinewidth=1, capsize=1.5)
    
    if show_fit: #Add the prediction/fit.
        ax.plot(objective.data.x, model, color="red", zorder=20)

    ax.set_xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
    ax.set_ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
    ax.set_yscale('log')
    ax.set_xlim(0, 0.3)
    ax.set_ylim(1e-7, 2)
    plt.show()
    return fig

if __name__ == "__main__":
    from refnx.dataset  import ReflectDataset
    from refnx.analysis import Objective

    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1,  thin_layer_sample_2
    from structures import easy_sample, many_param_sample

    structure   = easy_sample
    angle_times = {0.7: (70, 5),
                   2.0: (70, 20)}

    model, data   = simulate_single_contrast(*structure(), angle_times)
    q, r, r_error = data[:,0], data[:,1], data[:,2]
    plot_objective(Objective(model, ReflectDataset([q, r, r_error])), show_fit=False)
