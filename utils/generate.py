import numpy as np
import matplotlib.pyplot as plt

from refnx.reflect  import ReflectModel

POINTS = 300
Q_MIN  = 0.005
Q_MAX  = 0.3
DQ     = 2
SCALE  = 1
BKG    = 1e-6

def generate_noiseless(structure):
    model = ReflectModel(structure, scale=SCALE, dq=DQ, bkg=BKG)
    q = np.logspace(np.log10(Q_MIN), np.log10(Q_MAX), POINTS)
    r = model(q)
    r_error = [1e-10]*POINTS
    return [q, r, r_error]

def generate_noisy_single(structure, noise_constant=5000):
    model = ReflectModel(structure, scale=SCALE, dq=DQ, bkg=BKG)

    q = np.logspace(np.log10(Q_MIN), np.log10(Q_MAX), POINTS)
    r_model = model(q)
    r_noisy, r_error, flux_density = add_noise(q, r_model, noise_constant=noise_constant)
    data = [q, r_noisy, r_error]

    vary_model(model)
    return model, data, r_model, flux_density

def generate_noisy_multiple(save_path, structures):
    models = []
    datasets = []

    for i, structure in enumerate(structures, 1):
        model = ReflectModel(structure, scale=SCALE, dq=DQ, bkg=BKG)
        models.append(model)

        q = np.logspace(np.log10(Q_MIN), np.log10(Q_MAX), POINTS)
        r, r_error, _ = add_noise(q, model(q))
        datasets.append([q, r, r_error])
        
        data = np.zeros((POINTS, 3))
        data[:,0] = q
        data[:,1] = r
        data[:,2] = r_error
        np.savetxt(save_path+"/dataset{}.dat".format(i), data, delimiter=",")

    return models, datasets

def vary_model(model):
    for component in model.structure.components[1:-1]: #Skip over Air/D20 and substrate
        sld_bounds = (component.sld.real.value*0.5, component.sld.real.value*1.5)
        component.sld.real.setp(vary=True, bounds=sld_bounds)

        thick_bounds = (component.thick.value*0.5, component.thick.value*1.5)
        component.thick.setp(vary=True, bounds=thick_bounds)

        #Set the SLD and thickness to arbitrary initial values (within their bounds).
        component.sld.real.value = np.random.uniform(*sld_bounds)
        component.thick.value    = np.random.uniform(*thick_bounds)

def add_noise(q, r, file="../resources/directbeam_noise.dat", noise_constant=5000, bkg_rate=5e-7):
    #Try to load the beam sample file: exit the function if not found.
    direct_beam = np.loadtxt(file, delimiter=',')[:, 0:2]
    flux_density = np.interp(q, direct_beam[:, 0], direct_beam[:, 1]) * noise_constant #Not all Q values are the same

    #Background signal always ADDs to the signal.
    #Sometimes background could be 0. In which case it does not contribute to the signal
    r = [r_point + max(np.random.normal(1, 0.5) * bkg_rate, 0) for r_point in r]

    r_noisy = []
    r_error = []
    for flux, r_point in zip(flux_density, r): #Beam interp against simulated reflectance.
        count = flux*r_point
        error_bar = np.sqrt(count)
        noisy_count = np.random.normal(loc=count, scale=error_bar) #Using beam interp
        r_noisy.append(noisy_count / flux)
        r_error.append(error_bar   / flux)

    return r_noisy, r_error, flux_density

def plot_objective(objective):
    fig = plt.figure(figsize=[9,7])
    ax = fig.add_subplot(111)

    y, y_err, model = objective._data_transform(model=objective.generative())
    # Add the data in a transformed fashion.
    ax.errorbar(objective.data.x, y, y_err, marker="o", ms=3, lw=0, elinewidth=1, capsize=1.5)
    #Add the prediction/fit
    ax.plot(objective.data.x, model, color="red", zorder=20)

    plt.xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
    plt.ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
    plt.yscale('log')
    plt.show()