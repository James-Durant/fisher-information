import numpy as np
import os

from refnx.dataset import ReflectDataset
from refnx.reflect import Slab, ReflectModel
from refnx.analysis import Parameter, Objective, GlobalObjective, CurveFitter

from simulate import simulate_single_contrast
from utils import vary_structure

from plotting import plot_sld_profile, plot_sld_profiles, save_plot
from plotting import plot_reflectivity_curve, plot_objectives

import numbers
from refnx.reflect.structure import Scatterer
from refnx.analysis.parameter import BaseParameter, Parameters

class ModelGen:
    """The ModelGen class contains all code relating random model generation.

    Class Attributes:
        sld_bounds (tuple): the range of values that layer SLDs can take.
        thick_bounds (tuple): the range of values that layer thicknesses can take.
        rough_bounds (tuple): the range of values that layer roughnesses can take.
        substrate_sld (float): the SLD of the substrate (Silicon).
        angle_times (dict): a dictionary of points and times for each measurement angle.

    """
    sld_bounds    = (-1,10)
    thick_bounds  = (20,1000)
    rough_bounds  = (2,8)
    substrate_sld = 2.047
    angle_times   = {0.7: (70, 5),
                     2.0: (70, 20)}

    @staticmethod
    def generate(num_samples, layers):
        """Generates `num_samples` models and data sets with given number of `layers`.

        Args:
            num_samples (int): the number of models to generate.
            layers (int): the number of layers for each model to be generated with.

        Returns:
            models_data (list): `generate_num` models and associated simulated data sets.

        """
        models_data = []
        for layers in range(num_samples):
            # Get a random structure and simulate an experiement using it.
            structure = vary_structure(ModelGen.random_structure(layers),
                                       vary_rough=True, bound_size=0.25)
            
            models_data.append(simulate_single_contrast(structure,
                                         ModelGen.angle_times, 
                                         include_counts=True))
        return models_data

    @staticmethod
    def random_structure(layers):
        """Generates a single random structure with desired number of `layers`.

        Args:
            layers (int): the number of layers for the structure to be generated with.

        Returns:
            (refnx.reflect.Structure): the randomly generated structure.

        """
        # The structure consists of air followed by each layer and then finally the substrate.
        structure = SLD(0, name='Air')
        for i in range(layers):
            structure = structure | ModelGen.make_component(substrate=False)

        return structure | ModelGen.make_component(substrate=True)

    @staticmethod
    def make_component(substrate):
        """Generates a single layer of a structure.

        Args:
            substrate (Boolean): whether the component is the substrate or not.

        Returns:
            (refnx.reflect.Component): the randomly generated layer.

        """
        if substrate:
            thickness = 0 # Substrate has 0 thickness in refnx.
            sld = ModelGen.substrate_sld
        else:
            # Select a random thickness and SLD.
            thickness = np.random.choice(np.arange(*ModelGen.thick_bounds, 5))
            sld = np.random.choice(np.arange(*ModelGen.sld_bounds, 0.1))

        # Select a random roughness for the layer.
        roughness = np.random.choice(np.arange(*ModelGen.rough_bounds, 0.5))
        return SLD(sld)(thickness, roughness)

class SLD(Scatterer):
    def __init__(self, value, name=''):
        super().__init__(name=name)

        self.imag = Parameter(0, name='%s - isld' % name)
        if isinstance(value, numbers.Real):
            self.real = Parameter(value.real, name='%s - sld' % name)
        elif isinstance(value, numbers.Complex):
            self.real = Parameter(value.real, name='%s - sld' % name)
            self.imag = Parameter(value.imag, name='%s - isld' % name)
        elif isinstance(value, SLD):
            self.real = value.real
            self.imag = value.imag
        elif isinstance(value, BaseParameter):
            self.real = value
        elif (
            hasattr(value, '__len__')
            and isinstance(value[0], BaseParameter)
            and isinstance(value[1], BaseParameter)
        ):
            self.real = value[0]
            self.imag = value[1]

        self._parameters = Parameters(name=name)
        self._parameters.extend([self.real, self.imag])

    def __repr__(self):
        return 'SLD([{real!r}, {imag!r}],' ' name={name!r})'.format(
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

class Bilayer:
    def fit(self, fit_bkg=False):
        global_objective = GlobalObjective(self.objectives)
        
        if fit_bkg:
            for model in self.models:
                model.bkg.setp(vary=True, bounds=(1e-7, 1e-5))

            global_objective.varying_parameters = (lambda: self.parameters + [model.bkg for model in self.models])
        
        else:
            global_objective.varying_parameters = lambda: self.parameters

        fitter = CurveFitter(global_objective)
        fitter.fit('differential_evolution')
    
        print('----------- Fitted Parameters -----------')
        for param in global_objective.varying_parameters():
            print('{0}: {1}'.format(param.name, param.value))
    
    def plot_sld_profiles(self, save_path):
        fig, ax = plot_sld_profiles(self.structures, self.distances)
        
        save_path = os.path.join(save_path, str(self))
        save_plot(fig, save_path, 'sld_profiles')
    
    def plot_objectives(self, save_path):
        fig, ax = plot_objectives(self.objectives)
        ax.set_xscale('log')
        
        save_path = os.path.join(save_path, str(self))
        save_plot(fig, save_path, 'fitted_reflectivity')

class SymmetricBilayer(Bilayer):
    def __init__(self):
        self.data_path = './data/symmetric_bilayer'
        self.scales = [0.677763, 0.645217, 0.667776]
        self.bkgs = [3.20559e-06, 2.05875e-06, 2.80358e-06]
        self.dq = 2
        self.names = ['Si-D2O', 'Si-DMPC-D2O', 'Si-DMPC-H2O']
        self.distances = np.linspace(-20, 95, 500)
        
        self.si_sld      = 2.073
        self.sio2_sld    = 3.41
        self.dmpc_hg_vol = 320.9
        self.dmpc_tg_vol = 783.3
        self.dmpc_hg_sl  = 6.41e-4
        self.dmpc_tg_sl  = -3.08e-4
        self.water_vol   = 30.4
        
        # Calculate the SLD of the tails
        self.tg_sld = (self.dmpc_tg_sl / self.dmpc_tg_vol) * 1e6
        
        self.si_rough      = Parameter(2,     'Si/SiO2 rough',          (1,8))
        self.sio2_thick    = Parameter(14.7,  'SiO2 thick',             (5,20))
        self.sio2_rough    = Parameter(2,     'SiO2/DMPC rough',        (1,8))
        self.sio2_solv     = Parameter(0.245, 'SiO2 solv',              (0,1))
        self.dmpc_apm      = Parameter(49.9,  'DMPC apm',               (30,60))
        self.bilayer_rough = Parameter(6.57,  'Bilayer rough',          (1,8))
        self.bilayer_solv  = Parameter(0.074, 'Bilayer solv',           (0,1))
        self.hg_waters     = Parameter(3.59,  'Headgroup bound waters', (0,20))
        
        self.parameters = [self.si_rough,
                           self.sio2_thick,
                           self.sio2_rough,
                           self.sio2_solv,
                           self.dmpc_apm,
                           self.bilayer_rough,
                           self.bilayer_solv,
                           self.hg_waters]
        
        for param in self.parameters:
            param.vary=True
        
        self.create_objectives()
    
    def create_objectives(self):
        d2o_sl  = 2e-4
        d2o_sld = 6.19
        h2o_sl  = -1.64e-5
        h2o_sld = -0.5227
        
        D2O = SLD(d2o_sld)
        H2O = SLD(h2o_sld)
    
        # Relate headgroup bound waters to scattering lengths and volumes
        hg_water_d2o_sl = self.hg_waters * d2o_sl
        hg_water_h2o_sl = self.hg_waters * h2o_sl
        hg_water_vol    = self.hg_waters * self.water_vol
    
        # Add that to the headgroup volumes and scattering lengths in both contrast
        vol_hg = self.dmpc_hg_vol + hg_water_vol
    
        dmpc_hg_sl_d2o = self.dmpc_hg_sl + hg_water_d2o_sl
        dmpc_hg_sl_h2o = self.dmpc_hg_sl + hg_water_h2o_sl
    
        # Calculate the SLD of the headgroup in both contrast cases
        sld_hg_d2o = (dmpc_hg_sl_d2o / vol_hg) * 1e6 # SLD = sum b / v
        sld_hg_h2o = (dmpc_hg_sl_h2o / vol_hg) * 1e6
    
        # Calculate the thickness from the headgroup volume over the lipid area per molecule
        hg_thick = vol_hg / self.dmpc_apm # Thickness = v / APM
        # Calculate the thickness of the tails
        tg_thick = self.dmpc_tg_vol / self.dmpc_apm
    
        substrate    = SLD(self.si_sld)
        sio2         = Slab(self.sio2_thick, self.sio2_sld, self.si_rough,      vfsolv=self.sio2_solv)
        inner_hg_d2o = Slab(hg_thick,        sld_hg_d2o,    self.sio2_rough,    vfsolv=self.bilayer_solv)
        outer_hg_d2o = Slab(hg_thick,        sld_hg_d2o,    self.bilayer_rough, vfsolv=self.bilayer_solv)
        inner_hg_h2o = Slab(hg_thick,        sld_hg_d2o,    self.sio2_rough,    vfsolv=self.bilayer_solv)
        outer_hg_h2o = Slab(hg_thick,        sld_hg_h2o,    self.bilayer_rough, vfsolv=self.bilayer_solv)
        tg           = Slab(tg_thick,        self.tg_sld,   self.bilayer_rough, vfsolv=self.bilayer_solv)
    
        si_D2O_structure      = substrate | sio2 | D2O(rough=self.sio2_rough)
        si_DMPC_D2O_structure = substrate | sio2 | inner_hg_d2o | tg | tg | outer_hg_d2o | D2O(rough=self.bilayer_rough)
        si_DMPC_H2O_structure = substrate | sio2 | inner_hg_h2o | tg | tg | outer_hg_h2o | H2O(rough=self.bilayer_rough)
    
        self.structures = [si_D2O_structure, si_DMPC_D2O_structure, si_DMPC_H2O_structure]
        
        for i, structure in enumerate(self.structures):
            structure.name = self.names[i]
            
        self.models = [ReflectModel(structure, scale=scale, bkg=bkg, dq=self.dq)
                       for structure, scale, bkg in list(zip(self.structures, self.scales, self.bkgs))]

        self.datasets = [ReflectDataset(os.path.join(self.data_path, '{}.dat'.format(name)))
                         for name in self.names]
        
        self.objectives = [Objective(model, data) 
                           for model, data in list(zip(self.models, self.datasets))]

    def using_contrast(self, contrast_sld):
        hg_sld = contrast_sld*0.27 + 1.98*0.73
        
        vol_hg = self.dmpc_hg_vol + self.hg_waters*self.water_vol
    
        hg_thick = vol_hg / self.dmpc_apm 
        tg_thick = self.dmpc_tg_vol / self.dmpc_apm
    
        substrate = SLD(self.si_sld)
        sio2      = Slab(self.sio2_thick, self.sio2_sld, self.si_rough,      vfsolv=self.sio2_solv)
        inner_hg  = Slab(hg_thick,        hg_sld,        self.sio2_rough,    vfsolv=self.bilayer_solv)
        outer_hg  = Slab(hg_thick,        hg_sld,        self.bilayer_rough, vfsolv=self.bilayer_solv)
        tg        = Slab(tg_thick,        self.tg_sld,   self.bilayer_rough, vfsolv=self.bilayer_solv)
        solution  = SLD(contrast_sld)(rough=self.bilayer_rough)
    
        return substrate | sio2 | inner_hg | tg | tg | outer_hg | solution

    def __str__(self):
        return 'symmetric_bilayer'

class AsymmetricBilayer(Bilayer):
    def __init__(self):
        self.data_path = './data/asymmetric_bilayer'
        self.contrast_slds = [6.14, 2.07, -0.56]
        self.scale = 0.8
        self.bkgs = [4.6e-6, 8.6e-6, 8.7e-6]
        self.dq = 4
        self.names = ['dPC_RaLPS_D2O', 'dPC_RaLPS_SMW', 'dPC_RaLPS_H2O']
        self.distances = np.linspace(-30, 110, 500)
        
        self.si_sld    =  2.07
        self.sio2_sld  =  3.41
        self.pc_hg_sld =  1.98
        self.dPC_tg    =  7.45
        self.hLPS_tg   = -0.37
        self.core_D2O  =  4.2
        self.core_H2O  =  2.01
        
        self.si_rough       = Parameter(5.5,    'Si/SiO2 rough',   (3,8))
        self.sio2_thick     = Parameter(13.4,   'SiO2 thick',      (10,30))
        self.sio2_rough     = Parameter(3.2,    'SiO2/DPPC rough', (2,5))
        self.sio2_solv      = Parameter(0.038,  'SiO2 solv',       (0,0.5))
        self.inner_hg_thick = Parameter(9.0,    'Inner HG thick',  (5,20))
        self.inner_hg_solv  = Parameter(0.39,   'Inner HG solv',   (0,1))
        self.bilayer_rough  = Parameter(4.0,    'Bilayer rough',   (0,12))
        self.inner_tg_thick = Parameter(16.7,   'Inner TG thick',  (10,20))
        self.outer_tg_thick = Parameter(14.9,   'Outer TG thick',  (10,20))
        self.tg_solv        = Parameter(0.0085, 'TG solv',         (0,1))
        self.core_thick     = Parameter(28.7,    'Core thick',     (0,50))
        self.core_solv      = Parameter(0.26,   'Core solv',       (0,1))
        
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
        
        self.inner_tg_sld = None 
        self.outer_tg_sld = None
        
        for param in self.parameters:
            param.vary=True
        
    def create_objectives(self):
        self.structures = [self.using_contrast(sld, name)
                           for sld, name in list(zip(self.contrast_slds, self.names))]

        self.models = [ReflectModel(structure, scale=self.scale, bkg=bkg, dq=self.dq)
                       for structure, bkg in list(zip(self.structures, self.bkgs))]

        self.datasets = [ReflectDataset(os.path.join(self.data_path, '{}.dat'.format(name)))
                         for name in self.names]

        self.objectives = [Objective(model, data)
                           for model, data in list(zip(self.models, self.datasets))]
        
    def using_contrast(self, contrast_sld, name=''):
        contrast_point = (contrast_sld + 0.56) / (6.35 + 0.56)
        
        core_sld = contrast_point*self.core_D2O + (1-contrast_point)*self.core_H2O

        substrate = SLD(self.si_sld)
        solution  = SLD(contrast_sld)(0, self.bilayer_rough)
    
        if self.inner_tg_sld == None or self.outer_tg_sld == None:
            raise RuntimeError('inner/outer tailgroup SLD not defined')
    
        sio2        = Slab(self.sio2_thick,     self.sio2_sld,     self.si_rough,      vfsolv=self.sio2_solv)
        inner_hg    = Slab(self.inner_hg_thick, self.pc_hg_sld,    self.sio2_rough,    vfsolv=self.inner_hg_solv)
        inner_tg    = Slab(self.inner_tg_thick, self.inner_tg_sld, self.bilayer_rough, vfsolv=self.tg_solv)
        outer_tails = Slab(self.outer_tg_thick, self.outer_tg_sld, self.bilayer_rough, vfsolv=self.tg_solv)
        core        = Slab(self.core_thick,     core_sld,          self.bilayer_rough, vfsolv=self.core_solv)
        
        structure = substrate | sio2 | inner_hg | inner_tg | outer_tails | core | solution
        structure.name = name
        return structure
    
class SingleAsymmetricBilayer(AsymmetricBilayer):
    def __init__(self):
        super().__init__()
        
        self.asym_value = Parameter(0.95, 'Asymmetric value', (0,1), True)
        self.parameters.append(self.asym_value)
            
        self.inner_tg_sld = SLD(self.asym_value*self.dPC_tg  + (1-self.asym_value)*self.hLPS_tg)
        self.outer_tg_sld = SLD(self.asym_value*self.hLPS_tg + (1-self.asym_value)*self.dPC_tg)
        
        self.create_objectives()
    
    def __str__(self):
        return 'single_asymmetric_bilayer'
        
class DoubleAsymmetricBilayer(AsymmetricBilayer):
    def __init__(self):
        super().__init__()
        
        self.inner_tg_pc = Parameter(0.95, 'Inner TG PC', (0,1), True)
        self.outer_tg_pc = Parameter(0.063, 'Outer TG PC', (0,1), True)
        self.parameters.append(self.inner_tg_pc)
        self.parameters.append(self.outer_tg_pc)
            
        self.inner_tg_sld = SLD(self.inner_tg_pc*self.dPC_tg + (1-self.inner_tg_pc)*self.hLPS_tg)
        self.outer_tg_sld = SLD(self.outer_tg_pc*self.dPC_tg + (1-self.outer_tg_pc)*self.hLPS_tg)
        
        self.create_objectives()

    def __str__(self):
        return 'double_asymmetric_bilayer'
    
def QCS_sample():
    """Creates the QCS (quartz, copper, silicon) sample for which data was measured.

    Returns:
        refnx.reflect.Structure: refnx representation of the measured structure.

    """
    air       = SLD(0, name='Air')
    layer1    = SLD(1.795, name='Layer 1 - Si')(thick=790.7, rough=24.5)
    layer2    = SLD(6.385, name='Layer 2 - Cu')(thick=297.9, rough=3.5)
    substrate = SLD(3.354, name='Substrate - Quartz')(thick=0, rough=12.9)
    return air | layer1 | layer2 | substrate

def easy_sample():
    air       = SLD(0, name='Air')
    layer1    = SLD(4, name='Layer 1')(thick=100, rough=2)
    layer2    = SLD(8, name='Layer 2')(thick=150, rough=2)
    substrate = SLD(2.047, name='Substrate')(thick=0, rough=2)
    return air | layer1 | layer2 | substrate

def thin_layer_sample_1():
    air       = SLD(0, name='Air')
    layer1    = SLD(4, name='Layer 1')(thick=200, rough=2)
    layer2    = SLD(6, name='Layer 2')(thick=6, rough=2)
    substrate = SLD(2.047, name='Substrate')(thick=0, rough=2)
    return air | layer1 | layer2 | substrate

def thin_layer_sample_2():
    air       = SLD(0, name='Air')
    layer1    = SLD(4, name='Layer 1')(thick=200, rough=2)
    layer2    = SLD(5, name='Layer 2')(thick=30, rough=6)
    layer3    = SLD(6, name='Layer 3')(thick=6, rough=2)
    substrate = SLD(2.047, name='Substrate')(thick=0, rough=2)
    return air | layer1 | layer2 | layer3 | substrate

def similar_sld_sample_1():
    air       = SLD(0,   name='Air')
    layer1    = SLD(0.9, name='Layer 1')(thick=80, rough=2)
    layer2    = SLD(1.0, name='Layer 2')(thick=50, rough=6)
    substrate = SLD(2.047, name='Substrate')(thick=0, rough=2)
    return air | layer1 | layer2 | substrate

def similar_sld_sample_2():
    air       = SLD(0,   name='Air')
    layer1    = SLD(3.0, name='Layer 1')(thick=50, rough=2)
    layer2    = SLD(5.5, name='Layer 2')(thick=30, rough=6)
    layer3    = SLD(6.0, name='Layer 3')(thick=35, rough=2)
    substrate = SLD(2.047, name='Substrate')(thick=0, rough=2)
    return air | layer1 | layer2 | layer3 | substrate

def many_param_sample():
    air       = SLD(0,   name='Air')
    layer1    = SLD(2.0, name='Layer 1')(thick=50, rough=6)
    layer2    = SLD(1.7, name='Layer 2')(thick=15, rough=2)
    layer3    = SLD(0.8, name='Layer 3')(thick=60, rough=2)
    layer4    = SLD(3.2, name='Layer 4')(thick=40, rough=2)
    layer5    = SLD(4.0, name='Layer 5')(thick=18, rough=2)
    substrate = SLD(2.047, name='Substrate')(thick=0, rough=2)
    return air | layer1 | layer2 | layer3 | layer4 | layer5 | substrate

STRUCTURES = [easy_sample, QCS_sample, similar_sld_sample_1, similar_sld_sample_2,
              thin_layer_sample_1, thin_layer_sample_2, many_param_sample]

BILAYERS = [SymmetricBilayer, SingleAsymmetricBilayer, DoubleAsymmetricBilayer]

if __name__ == '__main__':
    save_path = './results'

    # Plot the SLD profiles and model reflecitivty curves for all structures.
    for structure in STRUCTURES:
        fig, _ = plot_sld_profile(structure())
        save_plot(fig, os.path.join(save_path, structure.__name__), 'sld_profile')
        
        fig, _ = plot_reflectivity_curve(structure())
        save_plot(fig, os.path.join(save_path, structure.__name__), 'model_reflectivity')

    # Plot the SLD profiles and fitted reflecitivty curves for the bilayer models.
    for bilayer_class in BILAYERS:
        bilayer = bilayer_class()
        bilayer.plot_sld_profiles(save_path)
        bilayer.plot_objectives(save_path)
