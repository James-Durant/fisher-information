import numpy as np

from refnx.analysis import Objective, CurveFitter

from simulate import simulate
from utils import vary_structure

def usefulness(structure, angle_times):
    objective = Objective(*simulate(structure(), angle_times))
    vary_structure(objective.model.structure, random_init=True)
    
    CurveFitter(objective).fit('differential_evolution', verbose=False)
    
    covar = objective.covar()
    
    n = len(objective.varying_parameters())
    
    pearson_rs = [covar[i,j] / (np.sqrt(covar[i,i])*np.sqrt(covar[j,j]))
                  for i in range(n)
                      for j in range(i+1, n)]
    
    return 1 - np.mean(np.abs(pearson_rs))

if __name__ == "__main__":
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1, thin_layer_sample_2
    from structures import easy_sample, many_param_sample
    
    angle_times = {0.7: (70, 5), #Angle: (Points, Time)
                   2.0: (70, 20)}
    
    for structure in [similar_sld_sample_1, similar_sld_sample_2,
                      thin_layer_sample_1, thin_layer_sample_2,
                      easy_sample, many_param_sample]:
    
        val = usefulness(structure, angle_times)
        print('{0}: {1}'.format(structure.__name__, val))