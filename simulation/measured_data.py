import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append("../")

from refnx.dataset  import ReflectDataset
from refnx.reflect  import SLD, ReflectModel
from refnx.analysis import Objective

from simulation.simulate import run_experiment
from simulation.utils    import plot_objective
from information.utils   import Sampler

def QCS_sample():
    """Creates the QCS (Quartz, Copper, Silicon) sample for which the measured
       data was recorded.

    Returns:
        refnx.reflect.Structure: refnx representation of the measured structure.

    """
    air       = SLD(0,      name="Air")
    layer1    = SLD(1.795,  name="Layer 1 - Si")(thick=790.7,   rough=24.5)
    layer2    = SLD(6.385,  name="Layer 2 - Cu")(thick=297.9,   rough=3.5)
    substrate = SLD(3.354,  name="Substrate - Quartz")(thick=0, rough=12.9)
    return air | layer1 | layer2 | substrate

def sample_measured_data(data_path, dq, bkg, save_path):
    """Samples the data in the 'QCS_all.dat' file using MCMC and nested sampling.

    Args:
        data_path (string): path to the directory containing the 'QCS_all.dat' file.
        dq (float): value to use for the model's instrument resolution parameter.
        bkg (float): value to use for the model's background parameter.
        save_path (string): path to the directory to save the sampling corner plots to.

    """
    structure = QCS_sample()

    #Vary the SLD, thickness and roughness of each layer (skip air and substrate).
    for component in structure.components[1:-1]:
        sld_bounds   = (component.sld.real.value*0.5, component.sld.real.value*1.5)
        thick_bounds = (component.thick.value*0.5, component.thick.value*1.5)
        rough_bounds = (component.rough.value*0.5, component.rough.value*1.5)
        component.sld.real.setp(vary=True, bounds=sld_bounds)
        component.thick.setp(vary=True, bounds=thick_bounds)
        component.rough.setp(vary=True, bounds=rough_bounds)

    model = ReflectModel(structure, scale=1, dq=dq, bkg=bkg)
    data  = ReflectDataset(data_path)
    data.scale(np.max(data.y)) #Scale the data so that the maximum reflectivity is 1.

    objective = Objective(model, data)
    sampler   = Sampler(objective)

    #Sample using MCMC and nested sampling and save the corner plots.
    fig1, _ = sampler.sample_MCMC(verbose=True)
    fig1.savefig(save_path+"/measured_sample_MCMC.png", dpi=600)
    fig2, _ = sampler.sample_nested(verbose=True)
    fig2.savefig(save_path+"/measured_sample_nested.png", dpi=600)

def plot_measured_all(data_path, dq, bkg, save_path):
    """Plots the binned measured data (combined data for all angles).

    Args:
        data_path (string): path to the directory containing the 'QCS_all.dat' file.
        dq (float): value to use for the model's instrument resolution parameter.
        bkg (float): value to use for the model's background parameter.
        save_path (string): path to the directory to save the plot to.

    Returns:
        scale (float): experimental scale factor for the measured data.

    """
    structure = QCS_sample()

    #Define the model, load and scale the data.
    model = ReflectModel(structure, scale=1, dq=dq, bkg=bkg)
    data  = ReflectDataset(data_path)
    scale = np.max(data.y)
    data.scale(scale) #Scale the data so that the maximum reflectivity is 1.

    #Plot the fit of the measured dataset and save it.
    objective = Objective(model, data)
    fig = plot_objective(objective)
    fig.savefig(save_path+"/measured_fit.png", dpi=600)
    return scale

def simulate_measured_data(data_path, files, dq, bkg, scale, angles, time):
    """Loads each measured angle's data and simulates experiments for each angle.

    Args:
        data_path (string): path to directory containing the measured data.
        files (list): list of file names for each angle's data.
        dq (float): value to use for the model's instrument resolution parameter.
        bkg (float): value to use for the model's background parameter.
        scale (float): value to use for the model's experimental scale factor.
        angles (list): the angles used in measuring the data.
        time (float): measurement time for the measured data.

    Returns:
        measured (dict): dictionary of measured reflectivity data for each angle.
        simulated (dict): dictionary of simulated reflectivity data for each angle.

    """
    #Iterate over each measurement angle and corresponding file.
    measured, simulated = {}, {}
    for angle, file_name in list(zip(angles, files)):
        #Load the measured reflectivity data.
        data = np.loadtxt(data_path+"/"+file_name, delimiter='    ')
        q_measured, r_measured, r_error_measured = data[:,0], data[:,1], data[:,2]
        measured[angle] = [q_measured, r_measured, r_error_measured]

        #For simulation, we need the edges of the Q bins but the measured data
        #contains Q bin centres so approximate the edges.
        points = len(q_measured)
        temp = [q_measured[0]-(q_measured[1]-q_measured[0])/2] + q_measured.tolist()
        q_bins = [(temp[i] + temp[i+1]) / 2 for i in range(points)]
        q_bins += [q_measured[-1]+(q_measured[-1]-q_measured[-2])/2]

        #Define the model with correct scale and simulate.
        model = ReflectModel(QCS_sample(), dq=dq, bkg=0, scale=scale)
        q_simulated, r_simulated, r_error_simulated, _ = run_experiment(model, angle, points, time, bkg, q_bins)
        simulated[angle] = [q_simulated, r_simulated, r_error_simulated]

    return measured, simulated

def plot_angle_data(angle_data, data_type, save_path):
    """Creates a plot overlaying each data set of each measured/simulated angle.
    
    Args:
        angle_data (dict): dictionary of reflectivity data for each angle.
        data_type (string): either 'measured' or 'simulated'.
        save_path (string): path to directory to save the plot to.
        
    """
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax  = fig.add_subplot(111)

    #Iterate over each simulated/measured angle.
    for angle in angle_data:
        q, r, r_error = angle_data[angle]
        #Plot Q values vs. reflectivity with associated reflectivity error bars.
        ax.errorbar(q, r, r_error, marker="o", ms=3, lw=0, elinewidth=1,
                    capsize=1.5, label="{0}° {1}".format(str(angle), data_type))

    ax.set_xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
    ax.set_ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
    ax.set_yscale('log')
    ax.set_xlim(0, 0.18) #Only plot up to the point that the measured data goes up to.
    ax.set_ylim(1e-7, 2)
    ax.legend()
    fig.savefig(save_path+"/{}_reflectivity.png".format(data_type.lower()), dpi=600)

def similarity(measured, simulated):
    """Calculates a similarity metric between given `measured` and `simulated`
       reflectivity data.

    Args:
        measured (dict): dictionary of measured reflectivity data for each angle.
        simulated (dict): dictionary of simulated reflectivity data for each angle.

    Returns:
        similarity (float): the similarity between the given datasets.

    """
    #Iterate over each measured/simulated angle.
    squared_diffs = []
    for angle in measured:
        #Get the measured and simulated data for the angle.
        q_measured,  r_measured,  r_error_measured  = measured[angle]
        q_simulated, r_simulated, r_error_simulated = simulated[angle]

        #Apply an Anscombe transform to make the values approximately normal.
        r_measured   = 2*np.sqrt(np.array(r_measured) + 3/8)
        r_simulated  = 2*np.sqrt(np.array(r_simulated) + 3/8)

        r_error_measured  = 2*np.sqrt(np.array(r_error_measured) + 3/8)
        r_error_simulated = 2*np.sqrt(np.array(r_error_simulated) + 3/8)

        #Weight the simulated and measured reflectivity by their errors.
        #If the reflectivity error is 0, calculate the weight as 0.
        weight1 = np.divide(r_measured,  r_error_measured,  out=np.zeros_like(r_measured),  where=r_error_measured!=0)
        weight2 = np.divide(r_simulated, r_error_simulated, out=np.zeros_like(r_simulated), where=r_error_simulated!=0)
        
        #Record the squared difference between the weighted reflectivity values for the angle.
        squared_diffs.append(np.square(weight1-weight2))

    #Calculate the mean over the weighted squared differences for all points (over all angles).
    #Then take the square root of this value to get a Hotelling T-statistic.
    t_statistic = np.sqrt(np.mean(squared_diffs))
    return t_statistic

if __name__ == "__main__":
    save_path = "./results"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_path = "./data"
    dq  = 2.5
    bkg = 8e-7

    #sample_measured_data(data_path+"/QCS_all.dat", dq, bkg, save_path)
    scale = plot_measured_all(data_path+"/QCS_all.dat", dq, bkg, save_path)

    angles = [0.3, 0.4, 0.5, 0.6, 0.7, 2.0]
    measured_data = ["QCS_03_1uA.dat", "QCS_04_1uA.dat", "QCS_05_1uA.dat",
                     "QCS_06_1uA.dat", "QCS_07_1uA.dat", "QCS_20_1uA.dat"]

    measured, simulated = simulate_measured_data(data_path, measured_data, dq, bkg, scale, angles, time=1)
    plot_angle_data(measured, "Measured", save_path)
    plot_angle_data(simulated, "Simulated", save_path)

    similarity = similarity(measured, simulated)
    print(similarity)
