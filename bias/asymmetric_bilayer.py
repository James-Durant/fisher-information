import matplotlib.pyplot as plt
import numpy as np

from refnx.dataset import ReflectDataset
from refnx.reflect import Slab, ReflectModel
from refnx.analysis import Parameter, Objective, GlobalObjective, CurveFitter

from simulate import simulate
from utils import calc_FIM, Sampler

import numbers
from refnx.reflect.structure import Scatterer
from refnx.analysis.parameter import BaseParameter, Parameters

class SLD(Scatterer):
    def __init__(self, value, name=""):
        super().__init__(name=name)

        self.imag = Parameter(0, name="%s - isld" % name)
        if isinstance(value, numbers.Real):
            self.real = Parameter(value.real, name="%s - sld" % name)
        elif isinstance(value, numbers.Complex):
            self.real = Parameter(value.real, name="%s - sld" % name)
            self.imag = Parameter(value.imag, name="%s - isld" % name)
        elif isinstance(value, SLD):
            self.real = value.real
            self.imag = value.imag
        elif isinstance(value, BaseParameter):
            self.real = value
        elif (
            hasattr(value, "__len__")
            and isinstance(value[0], BaseParameter)
            and isinstance(value[1], BaseParameter)
        ):
            self.real = value[0]
            self.imag = value[1]

        self._parameters = Parameters(name=name)
        self._parameters.extend([self.real, self.imag])

    def __repr__(self):
        return "SLD([{real!r}, {imag!r}]," " name={name!r})".format(
            **self.__dict__
        )

    def __complex__(self):
        sldc = complex(self.real.value, self.imag.value)
        return sldc

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component
        """
        self._parameters.name = self.name
        return self._parameters

class AsymmetricBilayer:
    def __init__(self, double_asymmetric=False):
        self.si_sld    =  2.07
        self.sio2_sld  =  3.41
        self.pc_hg_sld =  1.98
        self.dPC_tg    =  7.45
        self.hLPS_tg   = -0.37
        self.core_D2O  =  4.2
        self.core_H2O  =  2.01
        
        self.si_rough       = Parameter(6.83,     'Si/SiO2 rough',   (3,8))
        self.sio2_thick     = Parameter(13.9,     'SiO2 thick',      (10,30))
        self.sio2_rough     = Parameter(3.65,     'SiO2/DPPC rough', (2,5))
        self.sio2_solv      = Parameter(0.0392,   'SiO2 solv',       (0,0.5))
        self.inner_hg_thick = Parameter(9.02,     'Inner HG thick',  (5,20))
        self.inner_hg_solv  = Parameter(0.377,    'Inner HG solv',   (0,1))
        self.bilayer_rough  = Parameter(4.71,     'Bilayer rough',   (0,12))
        self.inner_tg_thick = Parameter(16.0,     'Inner TG thick',  (10,20))
        self.outer_tg_thick = Parameter(14.8,     'Outer TG thick',  (10,20))
        self.tg_solv        = Parameter(4.88e-05, 'TG solv',         (0,1))
        self.core_thick     = Parameter(28.5,     'Core thick',      (0,50))
        self.core_solv      = Parameter(0.245,    'Core solv',       (0,1))
        
        self.parameters = [self.si_rough,
                           self.sio2_thick,
                           self.sio2_rough,
                           self.sio2_solv,
                           self.inner_hg_thick,
                           self.inner_hg_solv,
                           self.bilayer_rough,
                           self.inner_tg_thick,
                           self.outer_tg_thick,
                           self.tg_solv,
                           self.core_thick,
                           self.core_solv]
        
        if double_asymmetric:
            self.inner_tg_pc = Parameter(0.986,  'Inner TG PC', (0,1))
            self.outer_tg_pc = Parameter(0.0257, 'Outer TG PC', (0,1))
            self.parameters.append(self.inner_tg_pc)
            self.parameters.append(self.outer_tg_pc)
            
            self.inner_tg_sld = SLD(self.inner_tg_pc*self.dPC_tg + (1-self.inner_tg_pc)*self.hLPS_tg)
            self.outer_tg_sld = SLD(self.outer_tg_pc*self.dPC_tg + (1-self.outer_tg_pc)*self.hLPS_tg)
            
        else:
            self.asym_value = Parameter(0.93, 'Asymmetric value', (0,1))
            self.parameters.append(self.asym_value)
            
            self.inner_tg_sld = SLD(self.asym_value*self.dPC_tg  + (1-self.asym_value)*self.hLPS_tg)
            self.outer_tg_sld = SLD(self.asym_value*self.hLPS_tg + (1-self.asym_value)*self.dPC_tg)
        
        for param in self.parameters:
            param.vary=True
        
    def using_contrast(self, contrast_sld, name=''):
        contrast_point = (contrast_sld + 0.56) / (6.35 + 0.56)
        
        core_sld = contrast_point*self.core_D2O + (1-contrast_point)*self.core_H2O

        substrate = SLD(self.si_sld)
        solution  = SLD(contrast_sld)(0, self.bilayer_rough)
    
        sio2        = Slab(self.sio2_thick,     self.sio2_sld,     self.si_rough,      vfsolv=self.sio2_solv)
        inner_hg    = Slab(self.inner_hg_thick, self.pc_hg_sld,    self.sio2_rough,    vfsolv=self.inner_hg_solv)
        inner_tg    = Slab(self.inner_tg_thick, self.inner_tg_sld, self.bilayer_rough, vfsolv=self.tg_solv)
        outer_tails = Slab(self.outer_tg_thick, self.outer_tg_sld, self.bilayer_rough, vfsolv=self.tg_solv)
        core        = Slab(self.core_thick,     core_sld,          self.bilayer_rough, vfsolv=self.core_solv)
        
        structure = substrate | sio2 | inner_hg | inner_tg | outer_tails | core | solution
        structure.name = name
        return structure

def fit_bilayer(double_asymmetric=False, slds=[6.14, 2.07, -0.56], 
                dq=4, scale=0.8, bkgs=[4.6e-6, 8.6e-6, 8.7e-6],
                names=['dPC_RaLPS_D2O', 'dPC_RaLPS_SMW', 'dPC_RaLPS_H2O']):
    
    bilayer = AsymmetricBilayer(double_asymmetric=double_asymmetric)
    
    structures = [bilayer.using_contrast(sld, name) 
                  for sld, name in list(zip(slds, names))]
    
    models = [ReflectModel(structure, dq=dq, scale=scale, bkg=bkg)
              for structure, bkg in list(zip(structures, bkgs))]
    
    datasets = [ReflectDataset('./data/{}.dat'.format(name)) 
                for name in names]
    
    objectives = [Objective(model, data) 
                  for model, data in list(zip(models, datasets))]

    for model in models:
        model.bkg.setp(vary=True, bounds=(1e-6, 1e-5))
    
    global_objective = GlobalObjective(objectives)
    global_objective.varying_parameters = (lambda: bilayer.parameters + [model.bkg for model in models])
    CurveFitter(global_objective).fit('differential_evolution')

    print('----------- Fitted Parameters -----------')
    for param in global_objective.varying_parameters():
        print('{0}: {1}'.format(param.name, param.value))
        
    plot_sld_profiles(structures)   
    plot_objectives(objectives)

def plot_sld_profiles(structures):
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)
    
    for structure in structures:
        ax.plot(*structure.sld_profile())
        
    ax.set_xlabel("$\mathregular{Distance\ (\AA)}$", fontsize=11, weight='bold')
    ax.set_ylabel("$\mathregular{SLD\ (10^{-6} \AA^{-2})}$", fontsize=11, weight='bold')
    return fig

def plot_objectives(objectives):  
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)
    
    for objective in objectives:
        data = objective.data
        model = objective.model
        ax.errorbar(data.x, data.y, data.y_err,marker="o", ms=3, lw=0, elinewidth=1, capsize=1.5)
        ax.plot(data.x, model(data.x), color="red", zorder=20)

    ax.set_xlabel("$\mathregular{Q\ (Å^{-1})}$", fontsize=11, weight='bold')
    ax.set_ylabel('Reflectivity (arb.)',         fontsize=11, weight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-7, 2)
    return fig

def plot_FIM(contrasts, angle_times):
    bilayer = AsymmetricBilayer(double_asymmetric=False)
    
    information = []
    for contrast_sld in contrasts:
        model, data = simulate(bilayer.using_contrast(contrast_sld), angle_times, include_counts=True)

        q, r, r_error, counts = data[:,0], data[:,1], data[:,2], data[:,3]

        objective = Objective(model, ReflectDataset([q, r, r_error]))
        xi = objective.varying_parameters()
        g = calc_FIM([q], xi, counts, [model])
        
        information.append(np.diag(g))

    information = np.asarray(information)
    
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax  = fig.add_subplot(111)
    for i, param in enumerate(xi):
        ax.plot(contrasts, information[:,i], label=param.name)

    ax.set_xlabel("$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$", fontsize=11, weight='bold')
    ax.set_ylabel('Fisher Information', fontsize=11, weight='bold')
    ax.set_yscale('log')
    ax.legend()

def contrast_choice(contrasts, angle_times):
    FIM_sums, fitting_sums = [], []
    for x, contrast in enumerate(contrasts):
        bilayer = AsymmetricBilayer(double_asymmetric=False)
        xi = bilayer.parameters
            
        true_vals = np.asarray([param.value for param in xi])
        
        model, data = simulate(bilayer.using_contrast(contrast), angle_times, include_counts=True)
        q, counts = data[:,0], data[:,3]

        g = 1 / calc_FIM([q], xi, counts, [model])
        
        objective = Objective(model, ReflectDataset([data[:,0], data[:,1], data[:,2]]))

        fitter = CurveFitter(objective)
        fitter.fit('differential_evolution', verbose=False)

        fitted_vals = np.asarray([param.value for param in xi])
        cov = objective.covar()
        
        #g = (g / true_vals[:, np.newaxis]) / true_vals[np.newaxis, :]
        #cov = (cov / fitted_vals[:, np.newaxis]) / fitted_vals[np.newaxis, :]
        
        FIM_sums.append(np.sum(g))
        fitting_sums.append(np.sum(cov))
        
        print("{0}/{1}".format(x+1, len(contrasts)))
    
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)
    ax.plot(contrasts, FIM_sums, label='Inverse Fisher Information') 
    ax.plot(contrasts, fitting_sums, label='Fitting Covariance') 
    ax.set_xlabel("$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$", fontsize=11, weight='bold')
    ax.set_ylabel("Normalised Matrix Sum", fontsize=11, weight='bold')
    ax.legend()

if __name__ == "__main__":
    fit_bilayer(double_asymmetric=False)
    
    angle_times = {0.7: (70, 10),
                   2.0: (70, 40)}
    contrasts = np.arange(-0.56, 6.35, 0.05)
    plot_FIM(contrasts, angle_times)
    #contrast_choice(contrasts, angle_times)
