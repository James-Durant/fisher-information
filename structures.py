from refnx.reflect import SLD
import matplotlib.pyplot as plt

def simple_model_sample():
    air       = SLD(0, name="Air")
    layer1    = SLD(2.5, name="Layer 1")(thick=500, rough=2)
    layer2    = SLD(5.0, name="Layer 2")(thick=100, rough=2)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    structure = air | layer1 | layer2 | substrate
    return [structure]

def multiple_contrast_samples():
    air       = SLD(0,   name="Air")
    D2O       = SLD(6.3, name="D20")
    layer1    = SLD(2.0, name="Layer 1")(thick=50, rough=6)
    layer2    = SLD(1.7, name="Layer 2")(thick=15, rough=2)
    layer3    = SLD(0.8, name="Layer 3")(thick=60, rough=2)
    layer4    = SLD(3.2, name="Layer 4")(thick=40, rough=2)
    layer5    = SLD(4.0, name="Layer 5")(thick=18, rough=2)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    
    structure_air = substrate | layer1 | layer2 | layer3 | layer4 | layer5 | air
    structure_D2O = substrate | layer1 | layer2 | layer3 | layer4 | layer5 | D2O
    return structure_air, structure_D2O

def thin_layer_samples_1():
    air       = SLD(0, name="Air")
    layer1    = SLD(4, name="Layer 1")(thick=200, rough=2)
    layer2    = SLD(6, name="Layer 2")(thick=6,   rough=2)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    structure = air | layer1 | layer2 | substrate
    return [structure]

def thin_layer_samples_2():
    air       = SLD(0, name="Air")
    layer1    = SLD(4, name="Layer 1")(thick=200, rough=2)
    layer2    = SLD(5, name="Layer 2")(thick=30,  rough=6)
    layer3    = SLD(6, name="Layer 3")(thick=6,   rough=2)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    structure = air | layer1 | layer2 | layer3 | substrate
    return [structure]

def similar_sld_samples_1():
    air       = SLD(0,   name="Air")
    layer1    = SLD(0.9, name="Layer 1")(thick=80, rough=2)
    layer2    = SLD(1.0, name="Layer 2")(thick=50, rough=6)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    structure = air | layer1 | layer2 | substrate
    return [structure]

def similar_sld_samples_2():
    air       = SLD(0,   name="Air")
    layer1    = SLD(3.0, name="Layer 1")(thick=50, rough=2)
    layer2    = SLD(5.5, name="Layer 2")(thick=30, rough=6)
    layer3    = SLD(6.0, name="Layer 3")(thick=35, rough=2)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    structure = air | layer1 | layer2 | layer3 | substrate
    return [structure]

def many_param_samples():
    air       = SLD(0,   name="Air")
    layer1    = SLD(2.0, name="Layer 1")(thick=50, rough=6)
    layer2    = SLD(1.7, name="Layer 2")(thick=15, rough=2)
    layer3    = SLD(0.8, name="Layer 3")(thick=60, rough=2)
    layer4    = SLD(3.2, name="Layer 4")(thick=40, rough=2)
    layer5    = SLD(4.0, name="Layer 5")(thick=18, rough=2)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    structure = air | layer1 | layer2 | layer3 | layer4 | layer5 | substrate
    return [structure]

def sld_profile(structure, show=True):
    plt.plot(*structure.sld_profile())
    plt.ylabel('SLD /$10^{-6} \AA^{-2}$')
    plt.xlabel('distance / $\AA$')
    if show:
        plt.show()

if __name__ == "__main__":
    structure1, structure2 = multiple_contrast_samples()
    sld_profile(structure1, show=False)
    sld_profile(structure2)
    
    sld_profile(*thin_layer_samples_1())
    sld_profile(*thin_layer_samples_2())
    sld_profile(*similar_sld_samples_1())
    sld_profile(*similar_sld_samples_2())
    sld_profile(*many_param_samples())