import numpy as np
import time

from refnx.dataset  import ReflectDataset
from refnx.reflect  import SLD, ReflectModel
from refnx.analysis import Objective

sld_bounds = (1,7)
thick_bounds = (0,500)
rough_bounds = (1,4)

from structures import multiple_contrast_samples, thin_layer_samples_1, thin_layer_samples_2
from structures import similar_sld_samples_1, similar_sld_samples_2, many_param_samples

class Data:
    points = 300
    q_min  = 0.005
    q_max  = 0.3
    dq     = 2
    scale  = 1
    bkg    = 1e-6
    
    @staticmethod
    def generate(structure):
        model = ReflectModel(structure, scale=Data.scale, dq=Data.dq, bkg=Data.bkg)
        
        q = np.logspace(np.log10(Data.q_min), np.log10(Data.q_max), Data.points)
        r, r_error, flux_density = Data.__add_noise(q, model(q))      
        data = [q, r, r_error]
        
        Data.vary_model(model)

        return model, data, flux_density
    
    @staticmethod
    def vary_model(model):
        components = model.structure.components
        for i, component in enumerate(components[1:-1]): #Skip over Air/D20 and substrate
            #Set the SLD, thickness and roughness to arbitrary initial values (within their bounds).
            #component.sld.real.value = (sld_bounds[1]   - sld_bounds[0])   / 2
            #component.thick.value    = (thick_bounds[1] - thick_bounds[0]) / 2
            
            component.sld.real.setp(vary=True, bounds=sld_bounds)
            component.thick.setp(vary=True, bounds=thick_bounds)
            component.rough.setp(vary=True, bounds=rough_bounds)
            
    
    @staticmethod
    def __add_noise(q, r, file="./directbeam_noise.dat", constant=100, bkg_rate=5e-7):
        """Adds background and sample noise to given reflectivity data using the direct beam sample.
    
        Args:
            q (ndarray): the range of q (momentum transfer) values.
            r (ndarray): the range of r (reflectance) values.
            file (string): the file path to the directbeam_noise file.
            constant (int): the sample noise constant.
            bkg_rate (type): the background rate value.
    
        Returns:
            Reflectance values with noise added.
    
        """
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
    epsilon = parameter.value * 0.005 #0.5% step
    old = parameter.value
    
    x1 = parameter.value = old - epsilon #First point
    y1 = model(q_point) 
    
    x2 = parameter.value = old + epsilon #Second point
    y2 = model(q_point) #Get new r value with changed model
    
    parameter.value = old #Reset parameter
    return (y2-y1) / (x2-x1)  #Return the gradient
        

def calc_fisher(structure):
    start = time.time()
    model, data, flux = Data.generate(structure)
    objective = Objective(model, ReflectDataset(data))
    [q, r, r_error] = data
    
    M = np.diag(flux/r, k=0)
    xi = objective.varying_parameters()
    
    n = len(r)
    m = len(xi)
    J = np.zeros((n,m)) #n,m
    for i in range(n):
        for j in range(m):
            J[i,j] = gradient(model, xi[j], q[i])
    
    fisher_info = np.dot(np.dot(J.T, M), J)
    print(fisher_info, "\n")
    end = time.time()
    print("Calculation Time: {}".format(end-start))
    return fisher_info

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
    fisher_info = calc_fisher(*structure)