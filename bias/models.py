from refnx.reflect import SLD, LipidLeaflet
from refnx.analysis import Parameter

class Bilayer:
    def __init__(self):
        self.substrate_sld     = 2.073
        self.substrate_rough   = Parameter(2, vary=False, name="Si/SiO2 Roughness")
        self.sio2_sld          = 3.41
        self.sio2_thick        = Parameter(14.7073, vary=False, name="SiO2 Thickness")
        self.sio2_rough        = Parameter(2, vary=False, name="SiO2/DMPC Roughness")
        self.sio2_hydration    = 0.245273
        self.bilayer_rough     = Parameter(6.5693, vary=False, name="Bilayer Roughness")
        self.bilayer_hydration = 0.073696
        
        HG_bounds_waters  = 3.588
        DMPC_apm          = 49.8936
        DMPC_HG_vol       = 320.9
        DMPC_tails_vol    = 783.3
        water_vol         = 30.4
        DMPC_tails_SL     = -3.08e-4
        
        vol_HG = DMPC_HG_vol + HG_bounds_waters*water_vol
        self.HG_thick = Parameter(vol_HG / DMPC_apm, vary=True, name="Headgroup Thickness")
    
        self.tails_sld   = DMPC_tails_SL / DMPC_tails_vol * 1e6
        self.tails_thick = Parameter(DMPC_tails_vol / DMPC_apm, vary=True, name="Tailgroup Thickness")
        
    def using_contrast(self, contrast_sld):
        HG_sld = contrast_sld*0.27 + 1.98*0.73
        
        substrate = SLD(self.substrate_sld)
    
        sio2 = SLD(self.sio2_sld)(thick=self.sio2_thick, rough=self.substrate_rough)
        sio2.vfsolv.setp(self.sio2_hydration)
        
        substrate = SLD(self.substrate_sld)
        sio2      = SLD(self.sio2_sld)(self.sio2_thick, self.substrate_rough)
        inner_HG  = SLD(HG_sld)(self.HG_thick, self.sio2_rough)
        outer_HG  = SLD(HG_sld)(self.HG_thick, self.bilayer_rough)
        tails     = SLD(self.tails_sld)(self.tails_thick, self.bilayer_rough)
        solution  = SLD(contrast_sld)(rough=self.bilayer_rough)
    
        sio2.vfsolv.setp(self.sio2_hydration)
        inner_HG.vfsolv.setp(self.bilayer_hydration)
        outer_HG.vfsolv.setp(self.bilayer_hydration)
        tails.vfsolv.setp(self.bilayer_hydration)
    
        return substrate | sio2 | inner_HG | tails | tails | outer_HG | solution

    @property
    def parameters(self):
        #return [self.substrate_rough, self.sio2_thick, self.sio2_rough,
        #        self.bilayer_rough,   self.HG_thick,   self.tails_thick]
        return [self.HG_thick, self.tails_thick]

class DMPC_refnx:
    def __init__(self):
        self.substrate_rough = Parameter(4.60093,  name="Si/SiO2 Roughness")
        self.sio2_thick      = Parameter(12.5021,  name="SiO2 Thickness")
        self.sio2_rough      = Parameter(3,        name="SiO2/DMPC Roughness")
        self.bilayer_rough   = Parameter(3,        name="Bilayer Roughness")
        self.HG_thick        = Parameter(9.61555,  name="Headgroup Thickness")
        self.tails_thick     = Parameter(5.67973,  name="Tailgroup Thickness")
        self.DMPC_apm        = Parameter(57.0848,  name="DMPC APM")
        self.solv_rough      = Parameter(0.863971, name="DMPC/Solution Roughness")
        
    def using_contrast(self, contrast_sld):
        si_substrate  = SLD(2.07)
        sio2_layer    = SLD(3.47*(1-0.14397)+contrast_sld*0.14397)(self.sio2_thick, self.substrate_rough)
        inner_leaflet = LipidLeaflet(self.DMPC_apm, 0.000601, 319, self.HG_thick, -0.000292, 782, self.tails_thick, self.bilayer_rough, self.sio2_rough)
        outer_leaflet = LipidLeaflet(self.DMPC_apm, 0.000601, 319, self.HG_thick, -0.000292, 782, self.tails_thick, self.bilayer_rough, 0, reverse_monolayer=True)
        solution      = SLD(contrast_sld)(thick=0, rough=self.solv_rough)
    
        return si_substrate | sio2_layer | inner_leaflet | outer_leaflet | solution

    @property
    def parameters(self):
        return [self.substrate_rough, self.sio2_thick, #self.sio2_rough,
                self.bilayer_rough,   self.HG_thick,   self.tails_thick,
                self.DMPC_apm,        self.solv_rough]