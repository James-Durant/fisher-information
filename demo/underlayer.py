import numpy as np

from refnx.reflect import SLD, Slab, Structure
from refnx.analysis import Parameter

from utils import thickness_choice

class Bilayer:
    """Defines a model describing a symmetric bilayer."""

    def __init__(self) -> None:
        self.name = 'symmetric_bilayer'
        
        self.si_sld      = 2.073
        self.sio2_sld    = 3.41
        self.dmpc_hg_vol = 320.9
        self.dmpc_tg_vol = 783.3
        self.dmpc_hg_sl  = 6.41e-4
        self.dmpc_tg_sl  = -3.08e-4
        self.water_vol   = 30.4

        # Calculate the SLD of the tails
        self.tg_sld = (self.dmpc_tg_sl / self.dmpc_tg_vol) * 1e6

        # Define the free parameters of the model.
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

    def using_contrast(self, contrast_sld: float) -> Structure:
        """Creates a structure representing the bilayer measured using a
           contrast of given `contrast_sld`.

        Args:
            contrast_sld (float): SLD of contrast to use.

        Returns:
            refnx.reflect.Structure: structure in contrast of given SLD.

        """
        # Calculate the SLD of the headgroup with the given contrast SLD.
        hg_sld = contrast_sld*0.27 + 1.98*0.73

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

# Path to directory to save results to.
save_path = './results'

# Number of points and time for each angle to simulate.
angle_times = {2.3: (30, 4)} # Angle: (Points, Time)

contrast_sld = 6.35 # SLD of contrast being simulated.

# Range of underlayer thicknesses to calculate the FIM over.
thicknesses = np.arange(2, 21, 0.25)

# Investigate how the FIM changes with underlayer thickness.
thickness_choice(Bilayer(), thicknesses, contrast_sld, angle_times, save_path)
