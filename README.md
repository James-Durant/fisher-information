# fisher-information
A framework for applying the Fisher information metric (FIM) to neutron reflectometry.

## About the Project
The project contains two main sections: experiment simulation and FIM application for which the code can be found in the [simulation](/simulation) and [information](/information) directories respectively. The measured data results can be found [here](/simulations/results) and the FIM results found [here](/information/results).

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
To do
