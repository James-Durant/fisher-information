import sys
sys.path.append("../utils") # Adds higher directory to python modules path.

from refnx.dataset  import ReflectDataset
from refnx.reflect  import SLD, ReflectModel
from refnx.analysis import Objective, CurveFitter

from dynesty import NestedSampler
from multiprocessing import Pool, cpu_count

from generate import generate_noiseless, plot_objective, SCALE, DQ, BKG

from structures import thin_layer_sample_1, thin_layer_sample_2
from structures import similar_sld_sample_1, similar_sld_sample_2
from structures import easy_sample_1, many_param_sample

class ModelSelection:
    def __init__(self, layer_bounds=(1,5), sld_bounds=(1,8), thick_bounds=(5,800), rough_bounds=(2,8),
                 substrate_sld=2.047, default_sld=2, default_thick=200, default_rough=2):
        self.layers_range  = list(range(layer_bounds[0], layer_bounds[1]+1))
        self.sld_bounds    = sld_bounds 
        self.thick_bounds  = thick_bounds 
        self.rough_bounds  = rough_bounds
        self.substrate_sld = substrate_sld
        self.default_sld   = default_sld
        self.default_thick = default_thick 
        self.default_rough = default_rough
  
    def __create_model(self, layers):
        air = SLD(0, name="Air")
        structure = air
        for i in range(1, layers+1):
            layer = SLD(self.default_sld, name="Layer {}".format(i))(thick=self.default_thick, rough=self.default_rough)
            layer.thick.setp(vary=True, bounds=self.thick_bounds)
            layer.sld.real.setp(vary=True, bounds=self.sld_bounds)
            #layer.rough.setp(vary=True, bounds=self.rough_bounds)
            structure = structure | layer             
            
        substrate = SLD(self.substrate_sld, name="Substrate")(thick=0, rough=self.default_rough)
        #substrate.rough.setp(vary=True, bounds=self.rough_bounds)
        
        structure = structure | substrate
        return ReflectModel(structure, scale=SCALE, dq=DQ, bkg=BKG)
    
    def select_model_logl(self, dataset, method='L-BFGS-B'):
        data = ReflectDataset(dataset)
        logl = []
        max_logl = -float("inf")
        best_model = None
        
        for layers in self.layers_range:
            print(">>> Fitting {}-layer model".format(layers))
            model = self.__create_model(layers)
            
            objective = Objective(model, data)
            fitter = CurveFitter(objective)
            fitter.fit(method, verbose=False)
            plot_objective(objective)
            
            model_logl = objective.logl()
            logl.append(model_logl)
            if model_logl < max_logl:
                max_logl = model_logl
                best_model = model
           
        print(logl)
        return best_model
    
    def select_model_logz(self, dataset):
        data = ReflectDataset(dataset)
        logz = []
        max_logz = -float("inf")
        best_model = None
        
        for layers in self.layers_range:
            print(">>> Fitting {}-layer model".format(layers))
            model = self.__create_model(layers)
            self.objective = Objective(model, data)
            
            pool = Pool(cpu_count()-1)
            ndim = len(self.objective.varying_parameters())
            sampler = NestedSampler(self.logl, self.objective.prior_transform, ndim,
                                    pool=pool, queue_size=cpu_count())
            sampler.run_nested()
            pool.close()
            pool.join()
            
            plot_objective(self.objective)
            model_logz = sampler.results.logz[-1]
            logz.append(model_logz)
            if model_logz < max_logz:
                max_logz = model_logz
                best_model = model
           
        print(logz)
        return best_model

    def logl(self, x):
        #Update the model
        for i, parameter in enumerate(self.objective.varying_parameters()):
            parameter.value = x[i]
        return self.objective.logl()

        
if __name__ == "__main__":
    structure = easy_sample_1()
    data = generate_noiseless(*structure)
    model = ModelSelection(layer_bounds=(1,5)).select_model_logl(data)