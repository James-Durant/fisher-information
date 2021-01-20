# fisher-information
A framework for applying the Fisher information metric (FIM) to neutron reflectometry.

## About The Project
The code relating to experiment simluation can be found in the [simulation](/simulation) directory and contains the following code:
* [measured_data.py](/simulation/measured_data.py) - Fits [measured data](/simulation/data) to obtain a ground truth model and uses this model to compare the measured data to its simulated counterpart. The results fitting, sampling and simulation can be found [here](/simulation/results).
* [simulate.py](/simulation/simulate.py) - Contains the code for simulating single and multiple contrast experiments. Uses a [directbeam](/simulation/data/directbeam_wavelength.dat) file of incident neutron flux as a function of wavelength.
* [structures.py](/simulation/structures.py) - Defines a number of structures of varying complexity used for FIM applications. The SLD profiles and model reflectivity curves of these structures can be found [here](/information/results).
* [utils.py](/simulation/utils.py) - Contains various utility functions for plotting SLD profiles and reflectivity curves.

The code relating to FIM calculation and application can be found in the [information](/information) directory and contains the following code:
* [benchmark.py](/information/benchmark.py) - Benchmarks the calculation of parameter uncertanties using the MCMC sampling, nested sampling and the FIM approach. Also contains the code for random model generation used in benchmarking. The results of the benchmark can be found [here](/information/results/benchmark.txt).
* [bilayer.py](/information/bilayer.py) - To do
* [confidence_ellipses.py](/information/confidence_ellipses.py) - Plots the FIM confidence ellipses on the corner plot from both MCMC and nested sampling.
* [time_dependence.py](/information/time_dependence.py) - Compares traditional fitting errors and FIM errors on simulated data with increasing time.
* [utils.py](/information/time_dependence.py) - Contains code for FIM calculation and sampling using MCMC or nested sampling.
* [variances.py](/information/variances.py) - Compares the inverse FIM with variance in parameter estimation using traditional fitting methods.
The results from these files can be found [here](/information/results).

### Built With
* [`Python3`](https://www.python.org/)
* [`refnx`](https://refnx.readthedocs.io/en/latest/)
* [`dynesty`](https://dynesty.readthedocs.io/en/latest/)
* [`corner.py`](https://corner.readthedocs.io/en/latest/)

## Installation
1. To replicate the development environment with the [`Anaconda`](https://www.anaconda.com/products/individual) distribution, first create an empty conda environment by running: <br /> ```conda create --name FIM```

2. To activate the environment, run: ```conda activate FIM```

3. Install pip by running: ```conda install pip```

4. Run the following to install the required packages from the [requirements.txt](/requirements.txt) file: <br />
   ```pip install -r requirements.txt```

You should now be able to run the code.

## Contact
Jos Cooper     - jos.cooper@stfc.ac.uk\
James Durant   - james.durant@stfc.ac.uk\
Lucas Wilkins  - lucas@lucaswilkins.com

## Acknowledgements
To do
