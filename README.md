# fisher-information
A framework for determining the maximum information gain and optimising experimental design in neutron reflectometry using the Fisher information metric (FIM).

## About the Project
This repository contains the [figures](/figures), [code](/fisher-information), [data](/fisher-information/data) and additional [results](/fisher-information/results) for the following article:
> Durant, J. H., Wilkins, L., Butler, K., and Cooper J. F. K. (2021). Determining the maximum information gain and optimising experimental design in neutron reflectometry using the Fisher information. *arXiv*. <br /> https://arxiv.org/abs/2103.08973

Please cite this article if you intend on including elements of this work in your own publications.

If you are interested in applying the framework to experimental design problems of your own, see [demo](/demo).

## Installation
1. To replicate the development environment with the [`Anaconda`](https://www.anaconda.com/products/individual) distribution, first create an empty conda environment by running: ```conda create --name FIM python=3.8.3```

2. To activate the environment, run: ```conda activate FIM```

3. Install pip by running: ```conda install pip```

4. Run the following to install the required packages from the [requirements.txt](/requirements.txt) file: <br />
   ```pip install -r requirements.txt```

You should now be able to run the code. Please ensure you are running a version of Python >= 3.8.0 \
If you are running an old version of Anaconda, you may need to reinstall with a newer version for this.

## Contact
Jos Cooper - jos.cooper@stfc.ac.uk \
James Durant - james.durant@warwick.ac.uk \
Lucas Wilkins - lucas@lucaswilkins.com

## Acknowledgements
This work has been partially supported by the STFC Facilities Programme Fund through the ISIS Neutron and Muon Source, and Scientific Computing Department of Rutherford Appleton Laboratory, Science and Technology Facilities Council, and by the Wave 1 of The UKRI Strategic Priorities Fund under the EPSRC Grant EP/T001569/1, particularly the "AI for Science" theme within that grant and The Alan Turing Institute. We would also like to thank Luke Clifton for his assistance and expertise in fitting the DMPC data.

## Contributing
Contributions are gladly accepted and would be very much appreciated.

  1. Fork the project
  2. Create your feature branch (```git checkout -b feature/AmazingFeature```)
  3. Commit your changes (```git commit -m 'Add some AmazingFeature'```)
  4. Push to the branch (```git push origin feature/AmazingFeature```)
  5. Open a pull request

## License
Distributed under the GPL-3.0 License. See [license](/LICENSE) for more information.
