from refnx.reflect import SLD

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
    structure_air.name = "multiple_contrast_sample_air"
    structure_D2O.name = "multiple_contrast_sample_D2O"
    return structure_air, structure_D2O

def easy_sample():
    air       = SLD(0, name="Air")
    layer1    = SLD(4, name="Layer 1")(thick=100, rough=2)
    layer2    = SLD(8, name="Layer 2")(thick=150, rough=2)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    return air | layer1 | layer2 | substrate

def thin_layer_sample_1():
    air       = SLD(0, name="Air")
    layer1    = SLD(4, name="Layer 1")(thick=200, rough=2)
    layer2    = SLD(6, name="Layer 2")(thick=6,   rough=2)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    return air | layer1 | layer2 | substrate

def thin_layer_sample_2():
    air       = SLD(0, name="Air")
    layer1    = SLD(4, name="Layer 1")(thick=200, rough=2)
    layer2    = SLD(5, name="Layer 2")(thick=30,  rough=6)
    layer3    = SLD(6, name="Layer 3")(thick=6,   rough=2)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    return air | layer1 | layer2 | layer3 | substrate

def similar_sld_sample_1():
    air       = SLD(0,   name="Air")
    layer1    = SLD(0.9, name="Layer 1")(thick=80, rough=2)
    layer2    = SLD(1.0, name="Layer 2")(thick=50, rough=6)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    return air | layer1 | layer2 | substrate

def similar_sld_sample_2():
    air       = SLD(0,   name="Air")
    layer1    = SLD(3.0, name="Layer 1")(thick=50, rough=2)
    layer2    = SLD(5.5, name="Layer 2")(thick=30, rough=6)
    layer3    = SLD(6.0, name="Layer 3")(thick=35, rough=2)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    return air | layer1 | layer2 | layer3 | substrate

def many_param_sample():
    air       = SLD(0,   name="Air")
    layer1    = SLD(2.0, name="Layer 1")(thick=50, rough=6)
    layer2    = SLD(1.7, name="Layer 2")(thick=15, rough=2)
    layer3    = SLD(0.8, name="Layer 3")(thick=60, rough=2)
    layer4    = SLD(3.2, name="Layer 4")(thick=40, rough=2)
    layer5    = SLD(4.0, name="Layer 5")(thick=18, rough=2)
    substrate = SLD(2.047, name="Substrate")(thick=0, rough=2)
    return air | layer1 | layer2 | layer3 | layer4 | layer5 | substrate

STRUCTURES = [similar_sld_sample_1, similar_sld_sample_2,
              thin_layer_sample_1,  thin_layer_sample_2,
              easy_sample,          many_param_sample]

def plot_sld_profiles(save_path):
    """Plots the SLD profiles of all structures in this file.

    Args:
        save_path (string): path to the directory to save the SLD profiles to.

    """
    #Plot and save SLD profiles for each single contrast structure.
    for structure_func in STRUCTURES:
        structure = structure_func()
        fig, _ = plot_sld_profile(structure)
        save_plot(fig, save_path+"/"+structure_func.__name__, "sld_profile")

    #Plot the multiple contrast sample on the same axis.
    structure1, structure2 = multiple_contrast_sample()
    fig, ax = plot_sld_profile(structure1, colour="C0", label="Air")
    ax.plot(*structure2.sld_profile(), color="C1", label="D2O")
    ax.legend()
    save_plot(fig, save_path+"/multiple_contrast_sample", "sld_profile")

def plot_reflectivity_curves(save_path):
    """Plots the reflectivity curves of all structures in this file.

    Args:
        save_path (string): path to the directory to save the reflectivity curves to.

    """
    #Plot and save reflectivity curves for each single contrast structure.
    for structure_func in STRUCTURES:
        structure = structure_func()
        fig, _ = plot_reflectivity_curve(structure)
        save_plot(fig, save_path+"/"+structure_func.__name__, "model_reflectivity")

    #Plot and save reflectivity curves for the multiple contrast sample.
    structure1, structure2 = multiple_contrast_sample()
    fig1, _ = plot_reflectivity_curve(structure1)
    fig2, _ = plot_reflectivity_curve(structure2)
    save_plot(fig1, save_path+"/multiple_contrast_sample", "model_reflectivity_air")
    save_plot(fig2, save_path+"/multiple_contrast_sample", "model_reflectivity_D2O")

if __name__ == "__main__":
    from simulation.utils import plot_sld_profile, plot_reflectivity_curve, save_plot

    save_path = "../information/results"
    plot_sld_profiles(save_path)
    plot_reflectivity_curves(save_path)
