import matplotlib.pyplot as plt
import numpy as np

from refnx.reflect import SLD, ReflectModel
from refnx.analysis import Objective, CurveFitter

from simulate import simulate

class ModelSelection:
    def __init__(self, layer_bounds=(1,5), sld_bounds=(1,9), thick_bounds=(5,400), rough_bounds=(2,8),
                 substrate_sld=2.047, default_sld=2, default_thick=200, default_rough=2):
        self.layers_range  = list(range(layer_bounds[0], layer_bounds[1]+1))
        self.sld_bounds    = sld_bounds 
        self.thick_bounds  = thick_bounds 
        self.rough_bounds  = rough_bounds
        self.substrate_sld = substrate_sld
        self.default_sld   = default_sld
        self.default_thick = default_thick 
        self.default_rough = default_rough
  
    def create_model(self, layers):
        air = SLD(0, name="Air")
        structure = air
        for i in range(1, layers+1):
            layer = SLD(self.default_sld, name="Layer {}".format(i))(thick=self.default_thick, rough=self.default_rough)
            layer.thick.setp(vary=True, bounds=self.thick_bounds)
            layer.sld.real.setp(vary=True, bounds=self.sld_bounds)
            structure = structure | layer             
            
        substrate = SLD(self.substrate_sld, name="Substrate")(thick=0, rough=self.default_rough)
        
        structure = structure | substrate
        return ReflectModel(structure, scale=1, dq=2, bkg=1e-7)
    
    def select_model(self, data):
        objectives, logls, AICs, BICs = [], [], [], []
        for layers in self.layers_range:
            print(">>> Fitting {}-layer model".format(layers))
            model = self.create_model(layers)
            
            objective = Objective(model, data)
            CurveFitter(objective).fit('differential_evolution', verbose=False)
            
            objectives.append(objective)
            plot_objective(objective)

            logl = objective.logl()
            logls.append(logl)

            k = len(objective.varying_parameters())
            AICs.append(2*k - 2*logl)
            
            n = len(data)
            BICs.append(k*np.log(n) - 2*logl)
            
        print("Log-likelihood: {}-layer".format(np.argmax(logls)+1))  
          
        min_AIC = np.argmin(AICs) 
        print("AIC: {}-layer".format(min_AIC+1))
        
        min_BIC = np.argmin(BICs) 
        print("BIC: {}-layer".format(min_BIC+1))
        
        return objectives[min_AIC]
        
def plot_objective(objective, show_fit=True):
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)

    #Add the data.
    q = objective.data.x
    r, r_error, r_model = objective._data_transform(model=objective.generative())
    ax.errorbar(q, r, r_error, color="black", marker="o", ms=3, lw=0, elinewidth=1, capsize=1.5)

    if show_fit: #Add the prediction/fit.
        ax.plot(q, r_model, color="red", zorder=20)

    ax.set_xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
    ax.set_ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
    ax.set_yscale('log')
    ax.set_xlim(0, 0.3)
    ax.set_ylim(1e-7, 2)
    plt.show()
    
if __name__ == "__main__":
    from structures import thin_layer_sample_1, thin_layer_sample_2
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import easy_sample, many_param_sample
    
    angle_times = {0.7: (70, 50), #Angle: (Points, Time)
                   2.0: (70, 200)}

    _, data = simulate(easy_sample(), angle_times)
    model = ModelSelection(layer_bounds=(1,4)).select_model(data)
    #print(model)