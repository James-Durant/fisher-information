# Experiment Simulation
* [measured_data.py](/simulation/measured_data.py) - Fits a measured [dataset](/simulation/data) to obtain a ground truth model and uses the model to simulate the measured data.
* [simulate.py](/simulation/simulate.py) - Contains the code for simulating single and multiple contrast experiments. Uses a [directbeam](/simulation/data/directbeam_wavelength.dat) file of incident neutron flux as a function of wavelength.
* [structures.py](/simulation/structures.py) - Defines a number of structures of varying complexity for use in applying the FIM.
* [utils.py](/simulation/utils.py) - Contains various utility functions for plotting SLD profiles, reflectivity curves and fits.
