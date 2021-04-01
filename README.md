# fisher-information
A framework for determining the maximum information gain and optimising experimental design in neutron reflectometry using the Fisher information metric (FIM).

## About the Project
The code, results and data for this project can be found in the [fisher-information](/fisher-information) directory. 

Please cite our paper corresponding to this repositiory if you include it in your own publications.
Its full reference is:
    "Durant, J. H., Wilkins, L., Butler, K., and Cooper J. F. K. (2021). Determining the maximum
information gain and optimising experimental design in neutron reflectometry using the Fisher information. arXiv, https://arxiv.org/abs/2103.08973"

The figures presented in our paper are avaliable in the [figures](/figures) directory.

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
Jos Cooper - jos.cooper@stfc.ac.uk\
James Durant - james.durant@stfc.ac.uk\
Lucas Wilkins - lucas@lucaswilkins.com

## Acknowledgements
This work has been partially supported by the  STFC Facilities Programme Fund  through the ISIS Neutron and Muon Source, and Scientific Computing Department of Rutherford Appleton Laboratory, Science and Technology Facilities Council, and by the Wave 1 of The UKRI Strategic Priorities Fund under the EPSRC Grant EP/T001569/1, particularly the "AI for Science" theme within that grant and The Alan Turing Institute. We would also like to thank Luke Clifton for his assistance and expertise in fitting the DMPC data.
