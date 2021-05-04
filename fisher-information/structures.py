import numpy as np
import os

from typing import List

from refnx.dataset import ReflectDataset
from refnx.reflect import SLD, Slab, Structure, ReflectModel
from refnx.analysis import Parameter, Objective, GlobalObjective, CurveFitter

from plotting import plot_sld_profile, plot_sld_profiles, save_plot
from plotting import plot_reflectivity_curve, plot_objectives
from utils import Sampler

from simulate import AngleTimes
from simulate import simulate_single_contrast as simulate

class Bilayer:
    """Parent class for the symmetric and asymmetric bilayer classes."""

    def fit(self, fit_bkg: bool=False) -> None:
        """Fits the bilayer model using measured data.

        Args:
            fit_bkg (bool): whether to fit the models' backgrounds.

        """
        # Create a global objective containing objectives for each contrast.
        global_objective = GlobalObjective(self.objectives)

        if fit_bkg:
            for model in self.models: # Vary the background parameters.
                model.bkg.setp(vary=True, bounds=(1e-7, 1e-5))

            # Add the background parameters to the already varying parameters.
            func = lambda: (self.parameters + [model.bkg for model in self.models])
        else:
            func = lambda: self.parameters # Varying parameters of the bilayer.

        # Overwrite varying_parameters method with the parameters of interest.
        global_objective.varying_parameters = func

        # Fit and display the estimated parameter values.
        fitter = CurveFitter(global_objective)
        fitter.fit('differential_evolution')

        print('----------- Fitted Parameters -----------')
        for param in global_objective.varying_parameters():
            print('{0}: {1}'.format(param.name, param.value))

    def sample(self, contrasts: List[float], angle_times: AngleTimes,
               save_path: str, filename: str) -> None:
        """Samples the bilayer model using nested sampling on simulated data.

        Args:
            contrasts (list): contrast SLDs to be sampled on.
            angle_times (dict): number of points and times for each angle.
            save_path (str): path to directory to save corner plots to.
            filename (str): name of file to use when saving plots.

        """
        # Create objectives for each contrast to sample with.
        objectives = []
        for contrast in contrasts:
            # Simulate an experiment using the given contrast.
            model, data = simulate(self.using_contrast(contrast), angle_times)
            objectives.append(Objective(model, data))

        # Combine objectives into a single global objective.
        global_objective = GlobalObjective(objectives)
        global_objective.varying_parameters = lambda: self.parameters

        # Sample the objective using nested sampling.
        sampler = Sampler(global_objective)
        fig = sampler.sample_nested(dynamic=True)

        # Save the sampling corner plot.
        save_path = os.path.join(save_path, str(self))
        save_plot(fig, save_path, filename+'_nested_sampling')

    def plot_sld_profiles(self, save_path: str) -> None:
        """Plots the SLD profile of each measured contrast.

        Args:
            save_path (str): path to directory to save SLD plot to.

        """
        fig, ax = plot_sld_profiles(self.structures, self.distances)

        save_path = os.path.join(save_path, str(self))
        save_plot(fig, save_path, 'sld_profiles')

    def plot_objectives(self, save_path: str) -> None:
        """Plots the fitted objectives of each measured contrast.

        Args:
            save_path (str): path to directory to save plot to.

        """
        fig, ax = plot_objectives(self.objectives)
        ax.set_xscale('log')

        save_path = os.path.join(save_path, str(self))
        save_plot(fig, save_path, 'fitted_reflectivity')

class SymmetricBilayer(Bilayer):
    """Defines a model describing a symmetric bilayer.

    Attributes:
        data_path (str): path to directory containing measured data.
        scales (list): experimental scale factor for each measured contrast.
        bkgs (list): experimental backgrounds for each measured contrast.
        dq (float): instrument resolution used when measuring.
        names (list): labels for each measured contrast.
        distances (numpy.ndarray): SLD profile x-axis values.
        si_sld (float): SLD of silicon substrate.
        sio2_sld (float): SLD of silicon oxide.
        dmpc_hg_vol (float): headgroup volume of DMPC bilayer.
        dmpc_tg_vol (float): tailgroup volume of DMPC bilayer.
        dmpc_hg_sl (float): headgroup scattering length of DMPC bilayer.
        dmpc_tg_sl (float): tailgroup scattering length of DMPC bilayer.
        water_vol (float): water volume of measured system.
        tg_sld (float): tailgroup SLD of DMPC bilayer.
        si_rough (refnx.analysis.Parameter): silicon substrate roughness.
        sio2_thick (refnx.analysis.Parameter): silicon oxide thickness.
        sio2_rough (refnx.analysis.Parameter): silicon oxide roughness.
        sio2_solv (refnx.analysis.Parameter): silicon oxide hydration.
        dmpc_apm (refnx.analysis.Parameter): DMPC area per molecule.
        bilayer_rough (refnx.analysis.Parameter): bilayer roughness.
        bilayer_solv (refnx.analysis.Parameter): bilayer hydration.
        hg_waters (refnx.analysis.Parameter): headgroup bound waters.
        parameters (list): varying model parameters.
        structures (list): structure for each measured contrast.
        models (list): model for each measured contrast.
        datasets (list): dataset for each measured contrast.
        objectives (list): objective for each measured contrast.

    """
    def __init__(self) -> None:
        self.data_path = './data/symmetric_bilayer'
        self.scales = [0.677763, 0.645217, 0.667776]
        self.bkgs = [3.20559e-06, 2.05875e-06, 2.80358e-06]
        self.dq = 2
        self.names = ['Si-D2O', 'Si-DMPC-D2O', 'Si-DMPC-H2O']
        self.distances = np.linspace(-20, 95, 500)

        # Define known values.
        self.si_sld      = 2.073
        self.sio2_sld    = 3.41
        self.dmpc_hg_vol = 320.9
        self.dmpc_tg_vol = 783.3
        self.dmpc_hg_sl  = 6.41e-4
        self.dmpc_tg_sl  = -3.08e-4
        self.water_vol   = 30.4

        # Calculate the SLD of the tails.
        self.tg_sld = (self.dmpc_tg_sl / self.dmpc_tg_vol) * 1e6

        # Define the varying parameters of the model.
        self.si_rough      = Parameter(2,     'Si/SiO2 Roughness',      (1,8))
        self.sio2_thick    = Parameter(14.7,  'SiO2 Thickness',         (5,20))
        self.sio2_rough    = Parameter(2,     'SiO2/DMPC Roughness',    (1,8))
        self.sio2_solv     = Parameter(0.245, 'SiO2 Hydration',         (0,1))
        self.dmpc_apm      = Parameter(49.9,  'DMPC Area Per Molecule', (30,60))
        self.bilayer_rough = Parameter(6.57,  'Bilayer Roughness',      (1,8))
        self.bilayer_solv  = Parameter(0.074, 'Bilayer Hydration',      (0,1))
        self.hg_waters     = Parameter(3.59,  'Headgroup Bound Waters', (0,20))

        self.parameters = [self.si_rough,
                           self.sio2_thick,
                           self.sio2_rough,
                           self.sio2_solv,
                           self.dmpc_apm,
                           self.bilayer_rough,
                           self.bilayer_solv,
                           self.hg_waters]

        # Vary all of the parameters defined above.
        for param in self.parameters:
            param.vary=True

        self.create_objectives() # Load the measured data.

    def __str__(self) -> str:
        return 'symmetric_bilayer'

    def create_objectives(self) -> None:
        """Creates objectives corresponding to each measured contrast."""

        # Define scattering lengths and densities of D2O and H2O.
        d2o_sl  = 2e-4
        d2o_sld = 6.19
        h2o_sl  = -1.64e-5
        h2o_sld = -0.5227

        D2O = SLD(d2o_sld)
        H2O = SLD(h2o_sld)

        # Relate headgroup bound waters to scattering lengths and volumes.
        hg_water_d2o_sl = self.hg_waters * d2o_sl
        hg_water_h2o_sl = self.hg_waters * h2o_sl
        hg_water_vol    = self.hg_waters * self.water_vol

        # Add to the headgroup volumes and scattering lengths in both contrast.
        vol_hg = self.dmpc_hg_vol + hg_water_vol

        dmpc_hg_sl_d2o = self.dmpc_hg_sl + hg_water_d2o_sl
        dmpc_hg_sl_h2o = self.dmpc_hg_sl + hg_water_h2o_sl

        # Calculate the SLD of the headgroup in both contrast cases
        sld_hg_d2o = (dmpc_hg_sl_d2o / vol_hg) * 1e6 # SLD = sum b / v
        sld_hg_h2o = (dmpc_hg_sl_h2o / vol_hg) * 1e6

        # Calculate the thickness from the headgroup volume over the lipid area per molecule.
        hg_thick = vol_hg / self.dmpc_apm # Thickness = v / APM

        # Calculate the thickness of the tailgroup
        tg_thick = self.dmpc_tg_vol / self.dmpc_apm

        # Define the layers of the structure.
        substrate = SLD(self.si_sld)
        sio2 = Slab(self.sio2_thick, self.sio2_sld, self.si_rough, vfsolv=self.sio2_solv)

        inner_hg_d2o = Slab(hg_thick, sld_hg_d2o,  self.sio2_rough,    vfsolv=self.bilayer_solv)
        outer_hg_d2o = Slab(hg_thick, sld_hg_d2o,  self.bilayer_rough, vfsolv=self.bilayer_solv)
        inner_hg_h2o = Slab(hg_thick, sld_hg_d2o,  self.sio2_rough,    vfsolv=self.bilayer_solv)
        outer_hg_h2o = Slab(hg_thick, sld_hg_h2o,  self.bilayer_rough, vfsolv=self.bilayer_solv)
        tg           = Slab(tg_thick, self.tg_sld, self.bilayer_rough, vfsolv=self.bilayer_solv)

        # Structure corresponding to measuring the Si/D2O interface.
        si_D2O_structure = substrate | sio2 | D2O(rough=self.sio2_rough)

        # Two structures corresponding to each measured contrast.
        si_DMPC_D2O_structure = substrate | sio2 | inner_hg_d2o | tg | tg | outer_hg_d2o | D2O(rough=self.bilayer_rough)
        si_DMPC_H2O_structure = substrate | sio2 | inner_hg_h2o | tg | tg | outer_hg_h2o | H2O(rough=self.bilayer_rough)

        self.structures = [si_D2O_structure, si_DMPC_D2O_structure, si_DMPC_H2O_structure]

        # Label the structures.
        for i, structure in enumerate(self.structures):
            structure.name = self.names[i]

        # Define models using structures above.
        self.models = [ReflectModel(structure, scale=scale, bkg=bkg, dq=self.dq)
                       for structure, scale, bkg in list(zip(self.structures, self.scales, self.bkgs))]

        # Load the measured datasets.
        self.datasets = [ReflectDataset(os.path.join(self.data_path, '{}.dat'.format(name)))
                         for name in self.names]

        # Combine models and datasets into objectives that can be fitted.
        self.objectives = [Objective(model, data) for model, data in list(zip(self.models, self.datasets))]

    def using_contrast(self, contrast_sld: float) -> Structure:
        """Creates a structure representing the bilayer measured using a
           contrast of given `contrast_sld`.

        Args:
            contrast_sld (float): SLD of contrast to simulate.

        Returns:
            (refnx.reflect.Structure): structure in given contrast.

        """
        # Calculate the SLD of the headgroup with the given contrast SLD.
        hg_sld = contrast_sld*0.27 + 1.98*0.73

        # Calculate the headgroup and tailgroup thicknesses with the given contrast SLD.
        vol_hg = self.dmpc_hg_vol + self.hg_waters*self.water_vol
        hg_thick = vol_hg / self.dmpc_apm
        tg_thick = self.dmpc_tg_vol / self.dmpc_apm

        substrate = SLD(self.si_sld)
        solution  = SLD(contrast_sld)(rough=self.bilayer_rough)

        sio2     = Slab(self.sio2_thick, self.sio2_sld, self.si_rough,      vfsolv=self.sio2_solv)
        inner_hg = Slab(hg_thick,        hg_sld,        self.sio2_rough,    vfsolv=self.bilayer_solv)
        outer_hg = Slab(hg_thick,        hg_sld,        self.bilayer_rough, vfsolv=self.bilayer_solv)
        tg       = Slab(tg_thick,        self.tg_sld,   self.bilayer_rough, vfsolv=self.bilayer_solv)

        solution  = SLD(contrast_sld)(rough=self.bilayer_rough)

        return substrate | sio2 | inner_hg | tg | tg | outer_hg | solution

class AsymmetricBilayer(Bilayer):
    """Defines a model describing an asymmetric bilayer. This model can either
       be defined using single or double asymmetry: the implementation for
       which is provided in the classes that inherit from this parent class.

    Attributes:
        data_path (str): path to directory containing measured data.
        contrast_slds (list): SLDs of measured contrasts.
        scale (float): experimental scale factor.
        bkgs (list): experimental backgrounds for each measured contrast.
        dq (float): instrument resolution.
        names (list): labels for each measured contrast.
        distances (numpy.ndarray): SLD profile x-axis values.
        si_sld (float): SLD of silicon substrate.
        sio2_sld (float): SLD of silicon oxide.
        pc_hg_sld (float):
        dPC_tg (float):
        hLPS_tg (float):
        core_D2O (float):
        core_H2O (float):
        si_rough (refnx.analysis.Parameter): silicon substrate roughness.
        sio2_thick (refnx.analysis.Parameter): silicon oxide thickness.
        sio2_rough (refnx.analysis.Parameter): silicon oxide roughness.
        sio2_solv (refnx.analysis.Parameter): silicon oxide hydration.
        inner_hg_thick (refnx.analysis.Parameter): inner headgroup thickness.
        inner_hg_solv (refnx.analysis.Parameter): inner headgroup hydration.
        bilayer_rough (refnx.analysis.Parameter): bilayer roughness.
        inner_tg_thick (refnx.analysis.Parameter): inner tailgroup thickness.
        outer_tg_thick (refnx.analysis.Parameter): outer tailgroup thickness.
        tg_solv (refnx.analysis.Parameter): tailgroup hydration.
        core_thick (refnx.analysis.Parameter): core thickness.
        core_solv (refnx.analysis.Parameter): core hydration.
        parameters (list): varying model parameters.
        structures (list): structure for each measured contrast.
        models (list): model for each measured contrast.
        datasets (list): dataset for each measured contrast.
        objectives (list): objective for each measured contrast.

    """
    def __init__(self) -> None:
        self.data_path = './data/asymmetric_bilayer'
        self.contrast_slds = [6.14, 2.07, -0.56]
        self.scale = 0.8
        self.bkgs = [4.6e-6, 8.6e-6, 8.7e-6]
        self.dq = 4
        self.names = ['dPC_RaLPS_D2O', 'dPC_RaLPS_SMW', 'dPC_RaLPS_H2O']
        self.distances = np.linspace(-30, 110, 500)

        # Define known values.
        self.si_sld    =  2.07
        self.sio2_sld  =  3.41
        self.pc_hg_sld =  1.98
        self.dPC_tg    =  7.45
        self.hLPS_tg   = -0.37
        self.core_D2O  =  4.2
        self.core_H2O  =  2.01

        # Define the varying parameters of the model.
        self.si_rough       = Parameter(5.5,    'Si/SiO2 Roughness',         (3,8))
        self.sio2_thick     = Parameter(13.4,   'SiO2 Thickness',            (10,30))
        self.sio2_rough     = Parameter(3.2,    'SiO2/Bilayer Roughness',    (2,5))
        self.sio2_solv      = Parameter(0.038,  'SiO2 Hydration',            (0,0.5))
        self.inner_hg_thick = Parameter(9.0,    'Inner Headgroup Thickness', (5,20))
        self.inner_hg_solv  = Parameter(0.39,   'Inner Headgroup Hydration', (0,1))
        self.bilayer_rough  = Parameter(4.0,    'Bilayer Roughness',         (0,12))
        self.inner_tg_thick = Parameter(16.7,   'Inner Tailgroup Thickness', (10,20))
        self.outer_tg_thick = Parameter(14.9,   'Outer Tailgroup Thickness', (10,20))
        self.tg_solv        = Parameter(0.0085, 'Tailgroup Hydration',       (0,1))
        self.core_thick     = Parameter(28.7,   'Core Thickness',            (0,50))
        self.core_solv      = Parameter(0.26,   'Core Hydration',            (0,1))

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

        # Vary all of the parameters defined above.
        for param in self.parameters:
            param.vary=True

    def create_objectives(self) -> None:
        """Creates objectives corresponding to each measured contrast."""

        # Define structures for each contrast.
        self.structures = [self.using_contrast(sld, name)
                           for sld, name in list(zip(self.contrast_slds, self.names))]

        # Define models for the structures above.
        self.models = [ReflectModel(structure, scale=self.scale, bkg=bkg, dq=self.dq)
                       for structure, bkg in list(zip(self.structures, self.bkgs))]

        # Load the measured data for each contrast.
        self.datasets = [ReflectDataset(os.path.join(self.data_path, '{}.dat'.format(name)))
                         for name in self.names]

        # Combine models and datasets into objectives corresponding to each contrast.
        self.objectives = [Objective(model, data) for model, data in list(zip(self.models, self.datasets))]

    def using_contrast(self, contrast_sld: float, name: str='') -> Structure:
        """Creates a structure representing the bilayer measured using a
           contrast of given `contrast_sld`.

        Args:
            contrast_sld (float): SLD of contrast to simulate.
            name (str): label for structure.

        Returns:
            (refnx.reflect.Structure): structure in given contrast.

        """
        # Calculate core SLD with the given contrast SLD.
        contrast_point = (contrast_sld + 0.56) / (6.35 + 0.56)
        core_sld = contrast_point*self.core_D2O + (1-contrast_point)*self.core_H2O

        substrate = SLD(self.si_sld)
        solution  = SLD(contrast_sld)(0, self.bilayer_rough)

        # Make sure this method is not being called from this class (should be from child class).
        if not hasattr(self, 'inner_tg_sld') or not hasattr(self, 'outer_tg_sld'):
            raise RuntimeError('inner/outer tailgroup SLD not defined')

        sio2     = Slab(self.sio2_thick,     self.sio2_sld,     self.si_rough,      vfsolv=self.sio2_solv)
        inner_hg = Slab(self.inner_hg_thick, self.pc_hg_sld,    self.sio2_rough,    vfsolv=self.inner_hg_solv)
        inner_tg = Slab(self.inner_tg_thick, self.inner_tg_sld, self.bilayer_rough, vfsolv=self.tg_solv)
        outer_tg = Slab(self.outer_tg_thick, self.outer_tg_sld, self.bilayer_rough, vfsolv=self.tg_solv)
        core     = Slab(self.core_thick,     core_sld,          self.bilayer_rough, vfsolv=self.core_solv)

        structure = substrate | sio2 | inner_hg | inner_tg | outer_tg | core | solution
        structure.name = name
        return structure

class SingleAsymmetricBilayer(AsymmetricBilayer):
    """Defines a model describing an asymmetric bilayer defined by a single
       asymmetry value. Inherits all of the attributes of the parent class.

    Attributes:
        asym_value (refnx.analysis.Parameter): bilayer asymmetry.
        inner_tg_sld (refnx.reflect.SLD): inner tailgroup SLD.
        outer_tg_sld (refnx.reflect.SLD): outer tailgroup SLD.

    """
    def __init__(self) -> None:
        super().__init__() # Call parent class constructor.

        # Define the single asymmetry parameter.
        self.asym_value = Parameter(0.95, 'Asymmetry Value', (0,1), True)
        self.parameters.append(self.asym_value)

        # Use asymmetry to define inner and outer tailgroup SLDs.
        self.inner_tg_sld = SLD(self.asym_value*self.dPC_tg  + (1-self.asym_value)*self.hLPS_tg)
        self.outer_tg_sld = SLD(self.asym_value*self.hLPS_tg + (1-self.asym_value)*self.dPC_tg)

        self.create_objectives() # Load the measured data.

    def __str__(self) -> str:
        return 'single_asymmetric_bilayer'

class DoubleAsymmetricBilayer(AsymmetricBilayer):
    """Defines a model describing an asymmetric bilayer defined by two
       asymmetry values. Inherits all of the attributes of the parent class.

    Attributes:
        inner_tg_pc (refnx.analysis.Parameter): 1st bilayer asymmetry.
        outer_tg_pc (refnx.analysis.Parameter): 2nd bilayer asymmetry.
        inner_tg_sld (refnx.reflect.SLD): inner tailgroup SLD.
        outer_tg_sld (refnx.reflect.SLD): outer tailgroup SLD.

    """
    def __init__(self) -> None:
        super().__init__() # Call parent class constructor.

        # Define the two asymmetry parameters.
        self.inner_tg_pc = Parameter(0.95, 'Inner Tailgroup PC', (0,1), True)
        self.outer_tg_pc = Parameter(0.063, 'Outer Tailgroup PC', (0,1), True)
        self.parameters.append(self.inner_tg_pc)
        self.parameters.append(self.outer_tg_pc)

        # Use the asymmetry parameters to define inner and outer tailgroup SLDs.
        self.inner_tg_sld = SLD(self.inner_tg_pc*self.dPC_tg + (1-self.inner_tg_pc)*self.hLPS_tg)
        self.outer_tg_sld = SLD(self.outer_tg_pc*self.dPC_tg + (1-self.outer_tg_pc)*self.hLPS_tg)

        self.create_objectives() # Load the measured data.

    def __str__(self) -> str:
        return 'double_asymmetric_bilayer'

def QCS_sample() -> Structure:
    """Creates the QCS (quartz, copper, silicon) sample.

    Returns:
        (refnx.reflect.Structure): refnx representation of the measured sample.

    """
    air = SLD(0, name='Air')
    layer1 = SLD(1.795, name='Layer 1 - Si')(thick=790.7, rough=24.5)
    layer2 = SLD(6.385, name='Layer 2 - Cu')(thick=297.9, rough=3.5)
    substrate = SLD(3.354, name='Substrate - Quartz')(thick=0, rough=12.9)
    return air | layer1 | layer2 | substrate

def easy_sample() -> Structure:
    air = SLD(0, name='Air')
    layer1 = SLD(4, name='Layer 1')(thick=100, rough=2)
    layer2 = SLD(8, name='Layer 2')(thick=150, rough=2)
    substrate = SLD(2.047, name='Substrate')(thick=0, rough=2)
    return air | layer1 | layer2 | substrate

def thin_layer_sample_1() -> Structure:
    air = SLD(0, name='Air')
    layer1 = SLD(4, name='Layer 1')(thick=200, rough=2)
    layer2 = SLD(6, name='Layer 2')(thick=6, rough=2)
    substrate = SLD(2.047, name='Substrate')(thick=0, rough=2)
    return air | layer1 | layer2 | substrate

def thin_layer_sample_2() -> Structure:
    air = SLD(0, name='Air')
    layer1 = SLD(4, name='Layer 1')(thick=200, rough=2)
    layer2 = SLD(5, name='Layer 2')(thick=30, rough=6)
    layer3 = SLD(6, name='Layer 3')(thick=6, rough=2)
    substrate = SLD(2.047, name='Substrate')(thick=0, rough=2)
    return air | layer1 | layer2 | layer3 | substrate

def similar_sld_sample_1() -> Structure:
    air = SLD(0, name='Air')
    layer1 = SLD(0.9, name='Layer 1')(thick=80, rough=2)
    layer2 = SLD(1.0, name='Layer 2')(thick=50, rough=6)
    substrate = SLD(2.047, name='Substrate')(thick=0, rough=2)
    return air | layer1 | layer2 | substrate

def similar_sld_sample_2() -> Structure:
    air = SLD(0, name='Air')
    layer1 = SLD(3.0, name='Layer 1')(thick=50, rough=2)
    layer2 = SLD(5.5, name='Layer 2')(thick=30, rough=6)
    layer3 = SLD(6.0, name='Layer 3')(thick=35, rough=2)
    substrate = SLD(2.047, name='Substrate')(thick=0, rough=2)
    return air | layer1 | layer2 | layer3 | substrate

def many_param_sample() -> Structure:
    air = SLD(0, name='Air')
    layer1 = SLD(2.0, name='Layer 1')(thick=50, rough=6)
    layer2 = SLD(1.7, name='Layer 2')(thick=15, rough=2)
    layer3 = SLD(0.8, name='Layer 3')(thick=60, rough=2)
    layer4 = SLD(3.2, name='Layer 4')(thick=40, rough=2)
    layer5 = SLD(4.0, name='Layer 5')(thick=18, rough=2)
    substrate = SLD(2.047, name='Substrate')(thick=0, rough=2)
    return air | layer1 | layer2 | layer3 | layer4 | layer5 | substrate

STRUCTURES = [easy_sample, QCS_sample, many_param_sample,
              similar_sld_sample_1, similar_sld_sample_2,
              thin_layer_sample_1, thin_layer_sample_2]

BILAYERS = [SymmetricBilayer, SingleAsymmetricBilayer, DoubleAsymmetricBilayer]

if __name__ == '__main__':
    save_path = './results'

    # Plot the SLD profiles and model reflectivity curves for all structures in this file.
    for structure in STRUCTURES:
        fig, _ = plot_sld_profile(structure())
        save_plot(fig, os.path.join(save_path, structure.__name__), 'sld_profile')

        fig, _ = plot_reflectivity_curve(structure())
        save_plot(fig, os.path.join(save_path, structure.__name__), 'model_reflectivity')

    # Define contrasts, measurement times and angles for sampling.
    D2O, SMW, H2O = 6.36, 2.07, -0.56
    angle_times = {0.7: (70, 10),
                   2.0: (70, 40)}

    # Plot the SLD profiles, fitted reflectivity curves and corner plots
    # for the bilayer models in this file.
    for bilayer_class in BILAYERS:
        bilayer = bilayer_class()
        bilayer.plot_sld_profiles(save_path)
        bilayer.plot_objectives(save_path)

        # Sample on D2O, D2Ox2, D2O+H2O, and D2O+SMW+H2O datasets.
        bilayer.sample([D2O], angle_times, save_path, 'D2O')
        bilayer.sample([D2O, H2O], angle_times, save_path, 'D2O_H2O')
        bilayer.sample([D2O, D2O], angle_times, save_path, 'D2O_D2O')
        bilayer.sample([D2O, SMW, H2O], angle_times, save_path, 'D2O_SMW_H2O')
