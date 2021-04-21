import matplotlib.pyplot as plt
import numpy as np

from typing import Optional, List, Tuple
from numpy.typing import ArrayLike

from refnx.dataset import ReflectDataset
from refnx.reflect import SLD, Component, Structure, ReflectModel
from refnx.analysis import Parameter, Objective, CurveFitter

from dynesty import NestedSampler, DynamicNestedSampler
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc

from simulate import simulate_single_contrast

class Sampler:
    """Samples an objective using MCMC and nested sampling.

    Attributes:
        objective (refnx.analysis.Objective): objective to sample.
        ndim (int): number of free parameters of the objective.
        sampler_MCMC (refnx.analysis.CurveFitter): sampler for MCMC sampling.
        sampler_nested_static (dynesty.NestedSampler): static nested sampler.
        sampler_nested_dynamic (dynesty.DynamicNestedSampler): dynamic nested sampler.

    """
    def __init__(self, objective: Objective) -> None:
        self.objective = objective
        self.ndim = len(self.objective.varying_parameters())
        self.sampler_MCMC = CurveFitter(self.objective)

        self.sampler_nested_static = NestedSampler(self.logl,
                                                   self.objective.prior_transform,
                                                   self.ndim)

        self.sampler_nested_dynamic = DynamicNestedSampler(self.logl,
                                                           self.objective.prior_transform,
                                                           self.ndim)

    def sample_MCMC(self, burn: int=400, steps: int=30, nthin: int=100,
                    fit_first: bool=True, verbose: bool=True,
                    show_fig: bool=True) -> Optional[plt.Figure]:
        """Samples the objective using MCMC sampling.

        Args:
            burn (int): number of samples to use for the burn-in period.
            steps (int): number of steps to use for the main sampling stage.
            nthin (int): amount of thinning to use for the main sampling stage.
            fit_first (bool): whether to fit before sampling.
            verbose (bool): whether to display progress when sampling.
            show_fig (bool): whether to create and return a corner plot.

        Returns:
            matplotlib.pyplot.Figure, optional: MCMC sampling corner plot.

        """
        # Initially fit with differential evolution if requested.
        if fit_first:
           self.sampler_MCMC.fit('differential_evolution', verbose=verbose)

        # Burn-in period.
        self.sampler_MCMC.sample(burn, verbose=verbose)

        # Main sampling stage.
        self.sampler_MCMC.reset()
        self.sampler_MCMC.sample(steps, nthin=nthin, verbose=verbose)

        # Return the sampling corner plot if requested.
        if show_fig:
            return self.objective.corner()

    def sample_nested(self, dynamic: bool=False, verbose: bool=True,
                      show_fig: bool=True) -> Optional[plt.Figure]:
        """Samples the objective using static or dynamic nested sampling.

        Args:
            dynamic (bool): whether to use dynamic or static nested sampling.
            verbose (bool): whether to display progress when sampling.
            show_fig (bool): whether to create and return a corner plot.

        Returns:
            matplotlib.pyplot.Figure, optional: nested sampling corner plot.

        """
        # Sample using dynamic or nested sampling.
        if dynamic:
            # Weighting is entirely on the posterior (0 weight on evidence).
            self.sampler_nested_dynamic.run_nested(print_progress=verbose,
                                                   wt_kwargs={'pfrac': 1.0})
            results = self.sampler_nested_dynamic.results
        else:
            self.sampler_nested_static.run_nested(print_progress=verbose)
            results = self.sampler_nested_static.results

        # Calculate the parameter means.
        weights = np.exp(results.logwt - results.logz[-1])
        mean, _ = dyfunc.mean_and_cov(results.samples, weights)

        # Update objective to use mean parameter values.
        self.logl(mean)

        # Return the sampling corner plot if requested.
        if show_fig:
            fig, _ = dyplot.cornerplot(results, color='blue', quantiles=None,
                                       show_titles=True, max_n_ticks=3,
                                       truths=np.zeros(self.ndim),
                                       truth_color='black')

            # Label axes with parameter labels.
            axes = np.reshape(np.array(fig.get_axes()), (self.ndim, self.ndim))
            parameters = self.objective.varying_parameters()
            for i in range(1, self.ndim):
                for j in range(self.ndim):
                    if i == self.ndim-1:
                        axes[i,j].set_xlabel(parameters[j].name)
                    if j == 0:
                        axes[i,j].set_ylabel(parameters[i].name)

            axes[self.ndim-1, self.ndim-1].set_xlabel(parameters[-1].name)

            return fig

    def logl(self, x: ArrayLike) -> float:
        """Calculates the log-likelihood of the parameters `x` against the model.

        Args:
            x (numpy.ndarray): array of parameter values.

        Returns:
            float: log-likelihood of the parameters x.

        """
        # Update the model with given parameter values.
        for i, parameter in enumerate(self.objective.varying_parameters()):
            parameter.value = x[i]
        return self.objective.logl()

class ModelGenerator:
    """Contains code relating random model generation.

    Attributes:
        sld_bounds (tuple): range of values that layer SLDs can take.
        thick_bounds (tuple): range of values that layer thicknesses can take.
        rough_bounds (tuple): range of values that layer roughnesses can take.
        substrate_sld (float): SLD of the substrate.
        angle_times (dict): points and times for each measurement angle.

    """
    def __init__(self, sld_bounds: Tuple[float, float]=(-1,10),
                 thick_bounds: Tuple[float, float]=(20,1000),
                 rough_bounds: Tuple[float, float]=(2,8),
                 substrate_sld: float=2.047) -> None:

        self.sld_bounds = sld_bounds
        self.thick_bounds = thick_bounds
        self.rough_bounds = rough_bounds
        self.substrate_sld = substrate_sld
        self.angle_times = {0.7: (70, 5),
                            2.0: (70, 20)}

    def generate(self, num_samples: int, layers: int
                 ) -> List[Tuple[ReflectModel, ReflectDataset, ArrayLike]]:
        """Generates `num_samples` models and datasets with given
           number of `layers`.

        Args:
            num_samples (int): number of models to generate.
            layers (int): number of layers for each model to be generated with.

        Returns:
            models_data (list): `generate_num` models and associated datasets.

        """
        models_data = []
        for layers in range(num_samples):
            # Get a random structure and simulate an experiment using it.
            structure = vary_structure(self.random_structure(layers),
                                       vary_rough=True, bound_size=0.25)

            models_data.append(simulate_single_contrast(structure,
                                                        self.angle_times,
                                                        include_counts=True))
        return models_data

    def random_structure(self, layers: int) -> Structure:
        """Generates a single random structure with desired number of layers.

        Args:
            layers (int): number of layers for generated structures.

        Returns:
            refnx.reflect.Structure: randomly generated structure.

        """
        # Air followed by each layer and then finally the substrate.
        structure = SLD(0, name='Air')
        for i in range(layers):
            structure |= self.make_component(substrate=False)

        structure |= self.make_component(substrate=True)
        return structure

    def make_component(self, substrate: bool) -> Component:
        """Generates a single layer of a structure.

        Args:
            substrate (bool): whether the component is the substrate or not.

        Returns:
            refnx.reflect.Component: randomly generated layer.

        """
        if substrate:
            thickness = 0 # Substrate has 0 thickness in refnx.
            sld = self.substrate_sld
        else:
            # Select a random thickness and SLD.
            thickness = np.random.choice(np.arange(*self.thick_bounds, 1))
            sld = np.random.choice(np.arange(*self.sld_bounds, 0.05))

        # Select a random roughness for the layer.
        roughness = np.random.choice(np.arange(*self.rough_bounds, 0.25))
        return SLD(sld)(thickness, roughness)

def vary_structure(structure: Structure, random_init: bool=False,
                   bound_size: float=0.2, vary_sld: bool=True,
                   vary_thick: bool=True, vary_rough: bool=False) -> Structure:
    """Vary the parameters of each layer of a given `structure` and optionally,
       initialise these values to random values within their bounds.

    Args:
        structure (refnx.reflect.Structure): structure to vary.
        random_init (bool): whether to randomly initialise parameters.
        bound_size (float): size of the bounds to place on the parameters.
        vary_sld (bool): whether to vary the structure's layers' SLDs.
        vary_thick (bool): whether to vary structure's layers' thicknesses.
        vary_rough (bool): whether to vary structure's layers' roughnesses.

    Returns:
        refnx.reflect.Structure: a reference to the given structure.

    """
    # Skip over air and the substrate.
    for component in structure.components[1:-1]:
        # Vary each layers' SLD, thickness and roughness if requested.
        if vary_sld:
            sld_bounds = (component.sld.real.value*(1-bound_size),
                          component.sld.real.value*(1+bound_size))
            component.sld.real.setp(vary=True, bounds=sld_bounds)
            # Set parameter to an arbitrary initial value within its bounds.
            if random_init:
                component.sld.real.value = np.random.uniform(*sld_bounds)

        if vary_thick:
            thick_bounds = (component.thick.value*(1-bound_size),
                            component.thick.value*(1+bound_size))
            component.thick.setp(vary=True, bounds=thick_bounds)
            if random_init:
                component.thick.value = np.random.uniform(*thick_bounds)

        if vary_rough:
            rough_bounds = (component.rough.value*(1-bound_size),
                            component.rough.value*(1+bound_size))
            component.rough.setp(vary=True, bounds=rough_bounds)
            if random_init:
                component.rough.value = np.random.uniform(*rough_bounds)

    # Vary the substrate's roughness.
    if vary_rough:
        component = structure.components[-1]
        rough_bounds = (component.rough.value*(1-bound_size),
                        component.rough.value*(1+bound_size))
        component.rough.setp(vary=True, bounds=rough_bounds)
        if random_init:
            component.rough.value = np.random.uniform(*rough_bounds)

    return structure

def fisher_single_contrast(q: ArrayLike, xi: List[Parameter], counts: ArrayLike,
                           model: ReflectModel) -> ArrayLike:
    """Calculates the FIM matrix for a given `model`.

    Args:
        q (numpy.ndarray): array of Q values.
        xi (list): parameters representing each varying parameter.
        counts (numpy.ndarray): incident neutron counts for each Q value.
        model (refnx.reflect.ReflectModel): model for calculating gradients.

    Returns:
        numpy.ndarray: FIM matrix for the model and data.

    """
    n = len(q)
    m = len(xi)
    J = np.zeros((n,m))
    # Calculate the gradient of the model reflectivity with every model
    # parameter for every model data point.
    for i in range(n):
        for j in range(m):
            J[i,j] = gradient(model, xi[j], q[i])

    r = model(q) #Use model reflectivity values
    M = np.diag(counts/r, k=0)
    return np.dot(np.dot(J.T, M), J)

def fisher_multiple_contrasts(qs: List[ArrayLike], xi: List[Parameter],
                              counts: List[ArrayLike],
                              models: List[ReflectModel]) -> ArrayLike:
    """Calculates the Fisher information metric (FIM) matrix for a given list
       of `models` and set of parameters, `xi`.

    Args:
        qs (list): Q arrays corresponding to each contrast.
        xi (list): varying parameters.
        counts (list): incident neutron counts corresponding to each Q value.
        models (list): models to calculate gradients with.

    Returns:
        numpy.ndarray: FIM matrix for the given models and parameters.

    """
    n = sum(len(q) for q in qs) # Number of data points.
    m = len(xi) # Number of parameters.
    J = np.zeros((n,m))

    # Calculate the gradient of the model reflectivity with every model
    # parameter for every model data point.
    r_all = []
    start = 0
    for q, model in list(zip(qs, models)):
        for i in range(len(q)):
            for j in range(m):
                J[start+i,j] = gradient(model, xi[j], q[i])

        start += len(q)
        r_all.append(model(q)) # Use model reflectivity values.

    r = np.concatenate(r_all)
    M = np.diag(np.concatenate(counts) / r, k=0)
    return np.dot(np.dot(J.T, M), J)

def gradient(model: ReflectModel, parameter: Parameter, q_point: float,
             step: float=0.005) -> float:
    """Calculate two-point gradient of model reflectivity with model parameter.

    Args:
        model (refnx.reflect.ReflectModel): model to calculate gradient.
        parameter (refnx.analysis.Parameter): parameter to vary.
        q_point (float): Q value of the R point to calculate the gradient of.
        step (float): step size to take when calculating the gradient.

    Returns:
        float: two-point gradient.

    """
    old, step = parameter.value, parameter.value*step # 0.5% step by default.

    x1 = parameter.value = old - step # First point
    y1 = model(q_point) # Get new r value with altered model.

    x2 = parameter.value = old + step # Second point
    y2 = model(q_point)

    parameter.value = old # Reset parameter
    return (y2-y1) / (x2-x1) # Return the gradient

def get_ground_truths(structure: Structure) -> ArrayLike:
    """Gets the stucture's true values of the layers' thicknesses and SLDs.

    Args:
        structure (refnx.reflect.Structure): structure to get true values from.

    Returns:
        numpy.ndarray: thickness and SLDs of each layer of the structure.

    """
    # Get the true values of the layers' thicknesses and SLDs.
    true = []
    for component in structure()[1:-1]:
        true.append(component.thick.value)
        true.append(component.sld.real.value)

    return np.asarray(true)

def usefulness(objective: Objective) -> float:
    """Calculate a usefulness metric for a given objective. This metric is
       intended to provide a measure of how closely the FIM results will
       match traditional sampling methods.

    Args:
        objective (refnx.analysis.Objective): objective to calculate
                                              usefulness of.

    Returns:
        float: usefulness metric for the objective.

    """
    covar = objective.covar() # Calculate the fitting covariance matrix.
    n = len(objective.varying_parameters())

    # Calculate the Pearson correlation coefficient between every parameter pair.
    pearson_rs = [covar[i,j] / (np.sqrt(covar[i,i])*np.sqrt(covar[j,j]))
                  for i in range(n)
                      for j in range(i+1, n)]

    # Calculate the mean absolute Pearson correlation coefficient.
    # 1 minus so that most useful is 1 and least useful is 0.
    return 1 - np.mean(np.abs(pearson_rs))

def select_model(dataset: ReflectDataset, counts: ArrayLike,
                 layers: Tuple[int, int]=(1,5)) -> ReflectModel:
    """Selects the best model for a given dataset.

    Args:
        dataset (refnx.dataset.ReflectDataset): dataset to obtain a model for.
        counts (numpy.ndarray): neutron counts for each point in the dataset.
        layers (tuple): range of layers to search when selecting a model.

    Returns:
        refnx.reflect.ReflectModel: selected model for the data.

    """
    generator = ModelGenerator()

    # Iterate over the layers to consider for the model.
    objectives, logls, AICs, BICs, KICs = [], [], [], [], []
    for layer in range(layers[0], layers[1]+1):
        # Display progress.
        print('>>> Fitting {}-layer model'.format(layer))

        # Generate a random model with the current number of layers.
        model, _, _ = generator.generate(1, layer)[0]

        # Fit the model against the given dataset.
        objective = Objective(model, dataset)
        CurveFitter(objective).fit('differential_evolution', verbose=False)

        objectives.append(objective)

        logl = objective.logl() # Record the log-likelihood.
        logls.append(logl)

        xi = objective.varying_parameters()

        # Calculate the Akaike information criterion (AIC)
        k = len(xi)
        AICs.append(-2*logl + 2*k)

        # Calculate the Bayesian information criterion (BIC)
        n = len(dataset)
        BICs.append(-2*logl + k*np.log(n))

        # Calculate the Kashyap information criterion (KIC)
        logp = objective.logp()
        g = fisher_single_contrast(dataset.x, xi, counts, model)
        KICs.append(-2*logl - 2*logp + k*np.log(n/(2*np.pi)) +
                    np.log(np.linalg.det(g/n)))

    # Display the best model using each information criterion.
    print('\nLog-likelihood: {}-layer'.format(np.argmax(logls)+1))
    print('AIC: {}-layer'.format(np.argmin(AICs)+1))
    print('BIC: {}-layer'.format(np.argmin(BICs)+1))
    print('KIC: {}-layer'.format(np.argmin(KICs)+1))

    return objectives[np.argmin(AICs)] # Use the AIC by default.

if __name__ == '__main__':
    from simulate import simulate_single_contrast

    from structures import thin_layer_sample_1, thin_layer_sample_2
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import easy_sample, QCS_sample, many_param_sample
    from structures import STRUCTURES

    structure = easy_sample
    angle_times = {0.7: (70, 5),
                   2.0: (70, 20)}

    # Select a model for the simulated data of the chosen structure.
    _, dataset, counts = simulate_single_contrast(structure(), angle_times,
                                                  include_counts=True)
    model = select_model(dataset, counts, layers=(1,4))
    print()

    # Calculate the usefulness of all structures in the structures file.
    print('----------- Usefulness Metrics -----------')
    for structure in STRUCTURES:
        model, data = simulate_single_contrast(structure(), angle_times)
        objective = Objective(model, data)
        vary_structure(objective.model.structure, random_init=True)

        CurveFitter(objective).fit('differential_evolution', verbose=False)

        val = usefulness(objective)
        print('{0}: {1}'.format(structure.__name__, val))
