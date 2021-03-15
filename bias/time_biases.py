import matplotlib.pyplot as plt
import numpy as np

from refnx.analysis import Objective, CurveFitter

from simulate import simulate

def vary_structure(structure):
    for component in structure[1:-1]:
        sld_bounds = (component.sld.real.value*0.25, component.sld.real.value*1.25)
        thick_bounds = (component.thick.value*0.25, component.thick.value*1.25)

        component.sld.real.setp(vary=True, bounds=sld_bounds)
        component.thick.setp(vary=True, bounds=thick_bounds)

        component.sld.real.value = np.random.uniform(*sld_bounds)
        component.thick.value = np.random.uniform(*thick_bounds)

def time_biases(structure, angle_times, multipliers, n=100):
    true = []
    for component in structure()[1:-1]:
        true.append(component.thick.value)
        true.append(component.sld.real.value)
        
    true = np.asarray(true)

    biases = []
    for i, multipler in enumerate(multipliers):
        fitted_params = []
        for _ in range(n):
            new_angle_times = {angle: (angle_times[angle][0], angle_times[angle][1]*multipler) 
                               for angle in angle_times}
            
            objective = Objective(*simulate(structure(), new_angle_times))
            vary_structure(objective.model.structure)
    
            fitter = CurveFitter(objective)
            fitter.fit('differential_evolution', verbose=False)
            
            fitted_params.append([param.value for param in objective.varying_parameters()])

        bias = np.mean(fitted_params, axis=0) - true
        biases.append(bias)
        
        print("{0}/{1}".format(i+1, len(multipliers)))
    
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)
    
    biases = np.asarray(biases)
    for i, param in enumerate(objective.varying_parameters()):
        ax.plot(multipliers, biases[:,i], label=param.name)

    ax.set_xlabel("Time Multiplier", fontsize=11, weight='bold')
    ax.set_ylabel("Bias", fontsize=11, weight='bold')
    ax.legend()

if __name__ == "__main__":
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1,  thin_layer_sample_2
    from structures import easy_sample, many_param_sample
    
    structure   = easy_sample
    angle_times = {0.7: (70, 5), #Angle: (Points, Time)
                   2.0: (70, 20)}

    multipliers = np.arange(1, 10, 0.2)
    time_biases(structure, angle_times, multipliers, 50)
