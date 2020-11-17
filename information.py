import numpy as np
import matplotlib.pyplot as plt

from refnx.dataset  import ReflectDataset
from refnx.reflect  import ReflectModel
from refnx.analysis import Objective, CurveFitter

from structures import multiple_contrast_samples, thin_layer_samples_1, thin_layer_samples_2
from structures import similar_sld_samples_1, similar_sld_samples_2, many_param_samples

POINTS = 300
Q_MIN  = 0.005
Q_MAX  = 0.3
DQ     = 2
SCALE  = 1
BKG    = 1e-6
SLD_BOUNDS = (0,9)
THICK_BOUNDS = (0,800)
ROUGH_BOUNDS = (1,9)
    
def generate(structure):
    model = ReflectModel(structure, scale=SCALE, dq=DQ, bkg=BKG)
    
    q = np.logspace(np.log10(Q_MIN), np.log10(Q_MAX), POINTS)
    r_model = model(q)
    r_noisy, r_error, flux_density = add_noise(q, r_model)      
    data = [q, r_noisy, r_error]
    
    vary_model(model)
    return model, data, r_model, flux_density

def vary_model(model):
    for component in model.structure.components[1:-1]: #Skip over Air/D20 and substrate
        sld_bounds = (component.sld.real.value*0.25, component.sld.real.value*1.75)
        component.sld.real.setp(vary=True, bounds=sld_bounds)
        
        thick_bounds = (component.thick.value*0.25, component.thick.value*1.75)
        component.thick.setp(vary=True, bounds=thick_bounds)
    
        #Set the SLD and thickness to arbitrary initial values (within their bounds).
        component.sld.real.value = sld_bounds[1]
        component.thick.value    = thick_bounds[1]

def add_noise(q, r, file="./directbeam_noise.dat", constant=100, bkg_rate=5e-7):
    #Try to load the beam sample file: exit the function if not found.
    direct_beam = np.loadtxt(file, delimiter=',')[:, 0:2]
    flux_density = np.interp(q, direct_beam[:, 0], direct_beam[:, 1]) #Not all Q values are the same

    #Background signal always ADDs to the signal.
    #Sometimes background could be 0. In which case it does not contribute to the signal
    r = [r_point + max(np.random.normal(1, 0.5) * bkg_rate, 0) for r_point in r]
    
    r_noisy = []
    r_error = []
    for i, r_point in zip(flux_density, r): #Beam interp against simulated reflectance.
        normal_width = r_point * constant / i
        r_noisy.append(np.random.normal(loc=r_point, scale=normal_width)) #Using beam interp
        r_error.append(normal_width)
        
    return r_noisy, r_error, flux_density

def gradient(model, parameter, q_point):
    step = parameter.value * 0.005 #0.5% step
    old = parameter.value
    
    x1 = parameter.value = old - step #First point
    y1 = model(q_point) 
    
    x2 = parameter.value = old + step #Second point
    y2 = model(q_point) #Get new r value with changed model
    
    parameter.value = old #Reset parameter
    return (y2-y1) / (x2-x1)  #Return the gradient  

def fisher(structure):
    model, data, r, flux = generate(structure)
    q = data[0]
    
    objective = Objective(model, ReflectDataset(data))
    fitter = CurveFitter(objective)
    fitter.fit('differential_evolution', polish=False)
    fitter.fit('L-BFGS-B')
    
    xi = objective.varying_parameters()
    n = len(r)
    m = len(xi)
    J = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            J[i,j] = gradient(model, xi[j], q[i])
    
    M = np.diag(flux/r, k=0)
    g = np.dot(np.dot(J.T, M), J) 
    #errors = 1 / np.diag(g)
    plot_ellipses(m, g, xi)
    return g

def plot_ellipses(m, g, xi):
    _, axes = plt.subplots(m, m, figsize=(15,15), dpi=300)
    for i in range(m):
        for j in range(m):
            if i > j:
                conﬁdence_ellipse(g, j, i, xi[j], xi[i], axes[i,j], i == m-1, j == 0)
            else:
                axes[i,j].set_visible(False)
    plt.show()

def conﬁdence_ellipse(fisher, i, j, param1, param2, axis, show_xlabel, show_ylabel, k=2):
    g = [[fisher[i,i], fisher[i,j]], [fisher[j,i], fisher[j,j]]] #Is this right?
    
    x = []
    y = []
    for theta in np.arange(0, 2*np.pi, 0.001):
        X = np.array([np.sin(theta), np.cos(theta)])
        epsilon = k / np.sqrt(np.dot(np.dot(X, g), X.T))
        x.append(epsilon*np.sin(theta))
        y.append(epsilon*np.cos(theta))
        
    #x = np.array(x) + param1.value
    #y = np.array(y) + param2.value
    
    axis.plot(x,y)
    if show_xlabel:
        axis.set_xlabel(param1.name)
    if show_ylabel:
        axis.set_ylabel(param2.name)
        
if __name__ == "__main__": 
    """
    Functions for getting structures:
        multiple_contrast_samples
        thin_layer_samples_1
        thin_layer_samples_2
        similar_sld_samples_1
        similar_sld_samples_2
        many_param_samples
    """
    
    structure = thin_layer_samples_1()
    fisher_info = fisher(*structure)
    #plot_errors(*structure)