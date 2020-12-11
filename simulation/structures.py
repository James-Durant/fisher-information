import os
from refnx.reflect import SLD
import matplotlib.pyplot as plt

def multiple_contrast_sample():
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

def easy_sample():
    air       = SLD(0, name="Air")
    layer1    = SLD(4, name="Layer 1")(thick=100, rough=2)
    layer2    = SLD(8, name="Layer 2")(thick=150, rough=2)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    structure = air | layer1 | layer2 | substrate
    return [structure]

def thin_layer_sample_1():
    air       = SLD(0, name="Air")
    layer1    = SLD(4, name="Layer 1")(thick=200, rough=2)
    layer2    = SLD(6, name="Layer 2")(thick=6,   rough=2)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    structure = air | layer1 | layer2 | substrate
    return [structure]

def thin_layer_sample_2():
    air       = SLD(0, name="Air")
    layer1    = SLD(4, name="Layer 1")(thick=200, rough=2)
    layer2    = SLD(5, name="Layer 2")(thick=30,  rough=6)
    layer3    = SLD(6, name="Layer 3")(thick=6,   rough=2)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    structure = air | layer1 | layer2 | layer3 | substrate
    return [structure]

def similar_sld_sample_1():
    air       = SLD(0,   name="Air")
    layer1    = SLD(0.9, name="Layer 1")(thick=80, rough=2)
    layer2    = SLD(1.0, name="Layer 2")(thick=50, rough=6)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    structure = air | layer1 | layer2 | substrate
    return [structure]

def similar_sld_sample_2():
    air       = SLD(0,   name="Air")
    layer1    = SLD(3.0, name="Layer 1")(thick=50, rough=2)
    layer2    = SLD(5.5, name="Layer 2")(thick=30, rough=6)
    layer3    = SLD(6.0, name="Layer 3")(thick=35, rough=2)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    structure = air | layer1 | layer2 | layer3 | substrate
    return [structure]

def many_param_sample():
    air       = SLD(0,   name="Air")
    layer1    = SLD(2.0, name="Layer 1")(thick=50, rough=6)
    layer2    = SLD(1.7, name="Layer 2")(thick=15, rough=2)
    layer3    = SLD(0.8, name="Layer 3")(thick=60, rough=2)
    layer4    = SLD(3.2, name="Layer 4")(thick=40, rough=2)
    layer5    = SLD(4.0, name="Layer 5")(thick=18, rough=2)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    structure = air | layer1 | layer2 | layer3 | layer4 | layer5 | substrate
    return [structure]

def plot_all_sld_profiles(save_path):
    """Plots the SLD profile for all structures in this file.

    Args:
        save_path (string): path to the directory to save the SLD profiles to.

    """
    #Plot and save SLD profiles for each single contrast structure.
    for structure_func in [similar_sld_sample_1, similar_sld_sample_2,
                           thin_layer_sample_1,  thin_layer_sample_2,
                           easy_sample,          many_param_sample]:
        structure = structure_func()[0]
        plt.figure()
        plt.plot(*structure.sld_profile()) #Plot SLD profile.
        plt.xlabel("$\mathregular{Distance\ (\AA)}$")
        plt.ylabel("$\mathregular{SLD\ (10^{-6} \AA^{-2})}$")

        #Save the SLD profile figure.
        structure_save_path = save_path+structure_func.__name__
        if not os.path.exists(structure_save_path):
            os.makedirs(structure_save_path)
        plt.savefig(structure_save_path+"/sld_profile.png", dpi=600)

    #Plot the two contrasts on the same axis.
    structure1, structure2 = multiple_contrast_sample()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(*structure1.sld_profile())
    ax.plot(*structure2.sld_profile())
    ax.set_xlabel("$\mathregular{Distance\ (\AA)}$")
    ax.set_ylabel("$\mathregular{SLD\ (10^{-6} \AA^{-2})}$")
    fig.savefig(save_path+"multiple_contrast_sample/sld_profile.png", dpi=600)

if __name__ == "__main__":
    save_path = "../comparison/results/"
    plot_all_sld_profiles(save_path)
