import numpy as np
import matplotlib.pyplot as plt

from refnx.reflect import ReflectModel

DQ       = 2
SCALE    = 1
BKG      = 1e-6
BKG_RATE = 5e-7

def simulate_noisy(structure, angle, points, time, directbeam_file="../simulation/directbeam_wavelength.dat"):
    """Simulates an experiment on a given `structure` with added realistic noise.

    Args:
        structure (refnx.reflect.Structure): the structure to simulate the experiment on.
        angle (float): the measurement angle for the simulation.
        points (int): the number of points to use when binning.
        time (int): the amount of time to count for during the experiment.
        directbeam_file (string): the file path to the directbeam_wavelength.dat file.

    Returns:
        model (refnx.reflect.ReflectModel): a model for the simulated experiment.
        q_binned (numpy.ndarray): Q values in equally log-spaced bins.
        r (numpy.ndarray): noisy reflectivity for each Q value.
        r_errors (numpy.ndarray): errors in each reflectivity value.
        flux_binned (numpy.ndarray): flux associated with each Q value.

    """
    #Load the directbeam_wavelength.dat file.
    direct_beam  = np.loadtxt(directbeam_file, delimiter=',')
    wavelengths  = direct_beam[:,0] # 1st column is wavelength, 2nd is flux.
    #Adjust flux by the time constant and measurement angle.
    flux_density = direct_beam[:,1]*time*np.power(angle/0.3, 2)

    theta = angle*np.pi/180 #Covert angle from degrees into radians.
    q = 4*np.pi*np.sin(theta) / wavelengths #Calculate Q values.

    #Bin Q values in equally log-spaced bins using flux as weighting.
    log_q = np.logspace(np.log10(np.min(q)), np.log10(np.max(q)), points+1)
    flux_binned, q_bin_edges = np.histogram(q, log_q, weights=flux_density)

    #Get the bin centres.
    q_binned = [(q_bin_edges[i] + q_bin_edges[i+1]) / 2 for i in range(points)]
    r, r_errors = [], []

    model = ReflectModel(structure, scale=SCALE, dq=DQ, bkg=BKG)

    for i in range(points): #Iterate over the desired number of points (bins).
        q_point = q_binned[i]
        flux    = flux_binned[i]

        #Calculate the model reflectivity and add background noise.
        r_point = model(q_point) + max(np.random.normal(1, 0.5)*BKG_RATE, 0)

        count       = r_point*flux #Calcuate the count for this bin.
        error_bar   = np.sqrt(count) #Calcuate the noisy count and its error bar.
        noisy_count = np.random.normal(loc=count, scale=error_bar)

        r_noisy = noisy_count / flux #Convert from count space back to reflectivity.
        r_error = error_bar / flux

        r.append(r_noisy)
        r_errors.append(r_error)

    return model, q_binned, r, r_errors, flux_binned

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

def plot_objective(objective):
    """Plots the fit of a given `objective` against the objective's data.

    Args:
        objective (refnx.analysis.Objective): the objective to plot.

    """
    fig = plt.figure(figsize=[9,7])
    ax = fig.add_subplot(111)

    #Add the data in a transformed fashion.
    y, y_err, model = objective._data_transform(model=objective.generative())
    ax.errorbar(objective.data.x, y, y_err, marker="o", ms=3, lw=0, elinewidth=1, capsize=1.5)
    #Add the prediction/fit.
    ax.plot(objective.data.x, model, color="red", zorder=20)

    plt.xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
    plt.ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
    plt.yscale('log')
    plt.show()
    return fig

if __name__ == "__main__":
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1,  thin_layer_sample_2
    from structures import easy_sample, many_param_sample, multiple_contrast_sample

    ax  = plt.figure(figsize=[9,7]).add_subplot(111)
    ax.set_xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
    ax.set_ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
    ax.set_yscale('log')

    structures = easy_sample()
    points     = 120
    time       = 1
    angles     = [0.7, 2]

    #Simulate an experiment for each contrast and each measurement angle of that contrast.
    for structure in structures:
        for angle in angles:
            _, q, r, r_error, _ = simulate_noisy(structure, angle, points, time)
            ax.errorbar(q, r, r_error, marker="o", ms=3, lw=0, elinewidth=1, capsize=1.5)
    plt.show()
