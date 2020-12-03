import sys
sys.path.append("../utils") #Adds higher directory to python modules path.

import numpy as np

from refnx.dataset  import ReflectDataset
from refnx.analysis import Objective

from information import calc_FIM
from generate    import plot_objective
from experiment  import simulate
from structures  import thin_layer_sample_1, thin_layer_sample_2
from structures  import similar_sld_sample_1, similar_sld_sample_2
from structures  import easy_sample_1, many_param_sample, multiple_contrast_sample

def fisher(structure, angle, xi=None):
    q, r, r_error, flux, model = simulate(structure, angle, points=50, time=100)
    r_model = model(q)
    
    for component in model.structure.components[1:-1]: #Skip over Air/D20 and substrate
        sld_bounds = (component.sld.real.value*0.5, component.sld.real.value*1.5)
        component.sld.real.setp(vary=True, bounds=sld_bounds)

        thick_bounds = (component.thick.value*0.5, component.thick.value*1.5)
        component.thick.setp(vary=True, bounds=thick_bounds)

    objective = Objective(model, ReflectDataset([q,r,r_error]))
    #plot_objective(objective)
    
    if xi is None:
        xi = objective.varying_parameters()
    else:
        for i, param in enumerate(objective.varying_parameters()):
            param.value = xi[i]
    
    g = calc_FIM(q, r_model, xi, flux, model)
    return np.diag(g), xi

if __name__ == "__main__":
    init_contrast, new_contrast = multiple_contrast_sample()
    init_angle, new_angle = 0.7, 2.0
    
    init_fisher, xi = fisher(init_contrast, init_angle)
    print("Initial Fisher Information: ", init_fisher, "\n")

    #1st choice: same contrast, new angle
    new_fisher, _ = fisher(init_contrast, new_angle, xi)
    print("1st Choice Fisher Information", new_fisher, "\n")
    
    #2nd choice: new contrast, same angle
    new_fisher, _ = fisher(new_contrast, init_angle, xi)
    print("2nd Choice Fisher Information", new_fisher, "\n")
    
    #3nd choice: new contrast, new angle
    new_fisher, _ = fisher(new_contrast, new_angle, xi)
    print("3rd Choice Fisher Information", new_fisher, "\n")



