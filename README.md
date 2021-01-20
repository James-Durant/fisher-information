# fisher-information
A framework for applying the Fisher information metric (FIM) to neutron reflectometry.

## About the Project
### Built With
* [`Python3`](https://www.python.org/)
* [`refnx`](https://refnx.readthedocs.io/en/latest/)
* [`dynesty`](https://dynesty.readthedocs.io/en/latest/)
* [`corner.py`](https://corner.readthedocs.io/en/latest/)

### Simulation
Experiment simulation code can be found in the [simulation](/simulation) directory:
* [measured_data.py](/simulation/measured_data.py) - Fits [measured data](/simulation/data) to obtain a ground truth model and uses this model to compare the data to its simulated counterpart. The results fitting, sampling and simulation can be found [here](/simulation/results).
* [simulate.py](/simulation/simulate.py) - Contains the code for simulating single and multiple contrast experiments. Uses a [directbeam](/simulation/data/directbeam_wavelength.dat) file of incident neutron flux as a function of wavelength.
* [structures.py](/simulation/structures.py) - Defines a number of structures of varying complexity used for FIM applications.
* [utils.py](/simulation/utils.py) - Contains various utility functions for plotting SLD profiles and reflectivity curves.

### Information
FIM calculation and application code can be found in the [information](/information) directory:
* [benchmark.py](/information/benchmark.py) - Benchmarks parameter uncertainty calculation using the MCMC sampling, nested sampling and the FIM approach.
* [bilayer.py](/information/bilayer.py) - To do
* [confidence_ellipses.py](/information/confidence_ellipses.py) - Plots the FIM confidence ellipses on MCMC and nested sampling corner plots.
* [time_dependence.py](/information/time_dependence.py) - Compares fitting errors and FIM errors with increasing time.
* [utils.py](/information/time_dependence.py) - Contains code for FIM calculation and sampling using MCMC or nested sampling.
* [variances.py](/information/variances.py) - Compares the inverse FIM with the variance in parameter estimation from fitting methods.

Results can be found [here](/information/results).

## Installation
1. To replicate the development environment with the [`Anaconda`](https://www.anaconda.com/products/individual) distribution, first create an empty conda environment by running: <br /> ```conda create --name FIM```

2. To activate the environment, run: ```conda activate FIM```

3. Install pip by running: ```conda install pip```

4. Run the following to install the required packages from the [requirements.txt](/requirements.txt) file: <br />
   ```pip install -r requirements.txt```

You should now be able to run the code.

## Contact
Jos Cooper - jos.cooper@stfc.ac.uk\
James Durant - james.durant@stfc.ac.uk\
Lucas Wilkins - lucas@lucaswilkins.com

## Acknowledgements
To do
