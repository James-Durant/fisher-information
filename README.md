# fisher-information
A framework for determining the maximum information gain and optimising experimental design in neutron reflectometry using the Fisher information metric (FIM).

## About the Project
This repository contains the [figures](/figures), [code](/fisher-information), [data](/fisher-information/data) and additional [results](/fisher-information/results) for the following paper:
> "Durant, J. H., Wilkins, L., Butler, K., and Cooper J. F. K. (2021). Determining the maximum information gain and optimising experimental design in neutron reflectometry using the Fisher information. *arXiv*. <br /> https://arxiv.org/abs/2103.08973"

Please cite this paper if you intend on including elements of this work in your own publications.

### Built With
* [`Python3`](https://www.python.org/) (>= 3.8.0)
* [`refnx`](https://refnx.readthedocs.io/en/latest/)
* [`dynesty`](https://dynesty.readthedocs.io/en/latest/)
* [`corner.py`](https://corner.readthedocs.io/en/latest/)

## Installation
1. To replicate the development environment with the [`Anaconda`](https://www.anaconda.com/products/individual) distribution, first create an empty conda environment by running: <br /> ```conda create --name FIM```

2. To activate the environment, run: ```conda activate FIM```

3. Install pip by running: ```conda install pip```

4. Run the following to install the required packages from the [requirements.txt](/requirements.txt) file: <br />
   ```pip install -r requirements.txt```

You should now be able to run the code. Please ensure you are running a version of Python >= 3.8.0

## Contact
Jos Cooper - jos.cooper@stfc.ac.uk\
James Durant - james.durant@warwick.ac.uk\
Lucas Wilkins - lucas@lucaswilkins.com

## Acknowledgements
This work has been partially supported by the STFC Facilities Programme Fund through the ISIS Neutron and Muon Source, and Scientific Computing Department of Rutherford Appleton Laboratory, Science and Technology Facilities Council, and by the Wave 1 of The UKRI Strategic Priorities Fund under the EPSRC Grant EP/T001569/1, particularly the "AI for Science" theme within that grant and The Alan Turing Institute. We would also like to thank Luke Clifton for his assistance and expertise in fitting the DMPC data.
