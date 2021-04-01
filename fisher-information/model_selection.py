import numpy as np

from refnx.reflect import SLD, ReflectModel
from refnx.analysis import Objective, CurveFitter

from simulate import simulate_single_contrast
from utils import calc_FIM

from plotting import plot_objective

class ModelSelector:
    def __init__(self, layer_bounds=(1,5), sld_bounds=(1,9), thick_bounds=(5,400), rough_bounds=(1,7),
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
        air = SLD(0, name='Air')
        structure = air
        for i in range(1, layers+1):
            layer = SLD(self.default_sld, name='Layer {}'.format(i))(thick=self.default_thick, rough=self.default_rough)
            layer.thick.setp(vary=True, bounds=self.thick_bounds)
            layer.sld.real.setp(vary=True, bounds=self.sld_bounds)
            #layer.rough.setp(vary=True, bounds=self.rough_bounds)
            structure = structure | layer             
            
        substrate = SLD(self.substrate_sld, name='Substrate')(thick=0, rough=self.default_rough)
        
        structure = structure | substrate
        return ReflectModel(structure, scale=1, bkg=1e-7,  dq=2)
    
    def select_model(self, dataset, counts):
        objectives, logls, AICs, BICs, KICs = [], [], [], [], []
        for layers in self.layers_range:
            print('>>> Fitting {}-layer model'.format(layers))
            model = self.create_model(layers)
            
            objective = Objective(model, dataset)
            CurveFitter(objective).fit('differential_evolution', verbose=False)
            
            objectives.append(objective)
            plot_objective(objective)

            logl = objective.logl()
            logls.append(logl)
            
            xi = objective.varying_parameters()

            k = len(xi)
            AICs.append(-2*logl + 2*k)
            
            n = len(dataset)
            BICs.append(-2*logl + k*np.log(n))
            
            logp = objective.logp()
            g = calc_FIM(dataset.x, xi, counts, model)
            KICs.append(-2*logl - 2*logp + k*np.log(n/(2*np.pi)) + np.log(np.linalg.det(g/n)))
            
        print('\nLog-likelihood: {}-layer'.format(np.argmax(logls)+1))  
          
        min_AIC = np.argmin(AICs) 
        print('AIC: {}-layer'.format(min_AIC+1))
        
        min_BIC = np.argmin(BICs) 
        print('BIC: {}-layer'.format(min_BIC+1))
        
        min_KIC = np.argmin(KICs) 
        print('KIC: {}-layer'.format(min_KIC+1))
        
        return objectives[min_AIC]
    
if __name__ == '__main__':
    from structures import thin_layer_sample_1, thin_layer_sample_2
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import easy_sample, QCS_sample, many_param_sample
    
    structure = easy_sample
    angle_times = {0.7: (70, 5), #Angle: (Points, Time)
                   2.0: (70, 20)}

    _, dataset, counts = simulate_single_contrast(structure(), angle_times, include_counts=True)
    model = ModelSelector(layer_bounds=(1,4)).select_model(dataset, counts)
