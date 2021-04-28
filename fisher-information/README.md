# Guide
* [benchmark.py](/fisher-information/benchmark.py) - Benchmarks parameter uncertainty calculation using MCMC sampling, nested sampling and the FIM approach.
* [biases.py](/fisher-information/biases.py) - Investigates biases in fitting, and determines how these biases change with measurement time and contrast choice.
* [ellipses.py](/fisher-information/ellipses.py) - Compares FIM confidence ellipses to MCMC and nested sampling corner plots.
* [experimental_design.py](/fisher-information/experimental_design.py) - Explores how the FIM varies with measurement angle, contrast choice and underlayer thickness.
* [plotting.py](/fisher-information/plotting.py) - Contains miscellaneous code for plotting SLD profiles and reflectivity curves.
* [similarity.py](/fisher-information/similarity.py) - Quantitatively and qualitatively compares a measured [dataset](/fisher-information/data/QCS_sample) to its simulated counterpart.
* [simulate.py](/fisher-information/simulate.py) - Simulates experiments using a [directbeam](/fisher-information/data/directbeam) file of incident neutron flux as a function of wavelength.
* [structures.py](/fisher-information/structures.py) - Defines samples of varying complexity for use in applying the FIM.
* [time_dependence.py](/fisher-information/time_dependence.py) - Explores how fitting and FIM uncertainties change with increasing measurement time.
* [utils.py](/fisher-information/utils.py) - Contains miscellaneous code for calculating the FIM, MCMC and nested sampling, random model generation and more.
* [variances.py](/fisher-information/variances.py) - Compares the inverse FIM with the variance in parameter estimation obtained over many fits.
