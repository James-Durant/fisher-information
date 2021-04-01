import numpy as np

from refnx.reflect import ReflectModel
from refnx.analysis import CurveFitter

from dynesty import NestedSampler, DynamicNestedSampler
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc

from structures import ModelGenerator

class Sampler:
    """The Sampler class contains code for MCMC and nested sampling.

    Attributes:
        objective (refnx.analysis.Objective): the objective to sample.
        ndim (int): the number of free parameters of the objective.
        sampler_MCMC (refnx.analysis.CurveFitter): refnx curve fitter for MCMC sampling.
        sampler_nested_static (dynesty.NestedSampler): dynesty static nested sampler.
        sampler_nested_dynamic (dynesty.DynamicNestedSampler): dynesty dynamic nested sampler.

    """
    def __init__(self, objective):
        self.objective = objective
        self.ndim = len(self.objective.varying_parameters())

        self.sampler_MCMC = CurveFitter(self.objective)
        self.sampler_nested_static = NestedSampler(self.logl, self.objective.prior_transform, self.ndim)
        self.sampler_nested_dynamic = DynamicNestedSampler(self.logl, self.objective.prior_transform, self.ndim)

    def sample_MCMC(self, burn=400, steps=30, nthin=100,
                    fit_first=True, verbose=True, show_fig=True):
        """Samples the objective using MCMC sampling.

        Args:
            burn (int): number of samples to use for the burn-in period.
            steps (int): number of steps to use for the main sampling stage.
            nthin (int): amount of thinning to use for the main sampling stage.
            fit_first (Boolean): whether to fit with differential evolution before sampling.
            verbose (Boolean): whether to display progress when sampling.
            show_fig (Boolean): whether to create and return a corner plot.

        Returns:
            (matplotlib.pyplot.Figure, optional): a MCMC sampling corner plot.

        """
        # Initially fit with differential evolution if requested.
        if fit_first:
           self.sampler_MCMC.fit('differential_evolution', verbose=verbose)

        self.sampler_MCMC.sample(burn, verbose=verbose) # Burn-in period.
        self.sampler_MCMC.reset()
        self.sampler_MCMC.sample(steps, nthin=nthin, verbose=verbose) # Main sampling stage.

        # Return the sampling corner plot if requested.
        if show_fig:
            return self.objective.corner()

    def sample_nested(self, dynamic=False, verbose=True, show_fig=True):
        """Samples the objective using static or dynamic nested sampling.

        Args:
            dynamic (Boolean): whether to use dynamic or static nested sampling.
            verbose (Boolean): whether to display progress when sampling.
            show_fig (Boolean): whether to create and return a corner plot.

        Returns:
            (matplotlib.pyplot.Figure, optional): nested sampling corner plot.

        """
        # Sample using dynamic or nested sampling.
        if dynamic:
            # Set weighting to be entirely on the posterior (0 weight on evidence).
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
            return dyplot.cornerplot(results, color='blue', quantiles=None,
                                     show_titles=True, max_n_ticks=3,
                                     truths=np.zeros(self.ndim),
                                     truth_color='black')[0]

    def logl(self, x):
        """Calculates the log-likelihood of the parameters x against the model.

        Args:
            x (numpy.ndarray): array of parameter values.

        Returns:
            float: log-likelihood of the parameters x.

        """
        # Update the model with given parameter values.
        for i, parameter in enumerate(self.objective.varying_parameters()):
            parameter.value = x[i]
        return self.objective.logl()

def vary_structure(structure, random_init=False, bound_size=0.2,
                   vary_sld=True, vary_thick=True, vary_rough=False):
    """Vary the parameters of each layer of a given structure and optionally,
       initialise these values to random values within their bounds.

    Args:
        structure (refnx.reflect.Structure): the structure to vary.
        random_init (Boolean): whether to randomly initialise the structure's parameters.
        bound_size (float): the size of the bounds to place on the parameters.
        vary_sld (Boolean): whether to vary the structure's layers' SLDs.
        vary_thick (Boolean): whether to vary the structure's layers' thicknesses.
        vary_rough (Boolean): whether to vary the structure's layers' roughnesses.

    Returns:
        refnx.reflect.Structure: a reference to the given structure.

    """
    # Skip over air and the substrate.
    for component in structure.components[1:-1]:
        # Vary each layers' SLD, thickness and roughness if requested.
        if vary_sld:
            sld_bounds = (component.sld.real.value*(1-bound_size), component.sld.real.value*(1+bound_size))
            component.sld.real.setp(vary=True, bounds=sld_bounds)
            # Set the parameter to an arbitrary initial value within its bounds.
            if random_init:
                component.sld.real.value = np.random.uniform(*sld_bounds)

        if vary_thick:
            thick_bounds = (component.thick.value*(1-bound_size), component.thick.value*(1+bound_size))
            component.thick.setp(vary=True, bounds=thick_bounds)
            if random_init:
                component.thick.value = np.random.uniform(*thick_bounds)

        if vary_rough:
            rough_bounds = (component.rough.value*(1-bound_size), component.rough.value*(1+bound_size))
            component.rough.setp(vary=True, bounds=rough_bounds)
            if random_init:
                component.rough.value = np.random.uniform(*rough_bounds)

    # Vary substrate's roughness.
    if vary_rough:
        component = structure.components[-1]
        rough_bounds = (component.rough.value*(1-bound_size), component.rough.value*(1+bound_size))
        component.rough.setp(vary=True, bounds=rough_bounds)
        if random_init:
            component.rough.value = np.random.uniform(*rough_bounds)

    return structure

def calc_FIM(qs, xi, counts, models):
    """Calculates the Fisher information metric (FIM) matrix for a given model
       and set of parameters.

    Args:
        q (numpy.ndarray or list): a single array of Q values or a list of Q arrays.
        xi (list): list of refnx Parameter objects representing each varying parameter.
        counts (numpy.ndarray or list): single array or list of arrays of incident neutron counts corresponding to each Q value.
        model (refnx.reflect.ReflectModel or list): the model or list of models to calculate the gradient with.

    Returns:
        numpy.ndarray: FIM matrix for the given model(s) and parameters.

    """
    # If given a single array, replace with a list containing the array.
    if not any(isinstance(x, np.ndarray) for x in qs):
        qs = [qs]

    # If given a list of counts, concatenate them to get a single array.
    if not any(isinstance(x, np.ndarray) for x in counts):
        counts = np.asarray(counts)
    else:
        counts = np.concatenate(counts)

    # If given a single model, replace with a list containing the model.
    if isinstance(models, ReflectModel):
        models = [models]

    n = sum(len(q) for q in qs) # Number of data points.
    m = len(xi) # Number of parameters.
    J = np.zeros((n,m))

    # Calculate the gradient of the model reflectivity with every model parameter for every model data point.
    r_all = []
    start = 0
    for q, model in list(zip(qs, models)):
        for i in range(len(q)):
            for j in range(m):
                J[start+i,j] = gradient(model, xi[j], q[i])

        start += len(q)
        r_all.append(model(q)) # Use model reflectivity values.

    r = np.concatenate(r_all)
    M = np.diag(counts / r, k=0)
    return np.dot(np.dot(J.T, M), J)

def gradient(model, parameter, q_point, step=0.005):
    """Calculate a two-point gradient of model reflectivity with model parameter.

    Args:
        model (refnx.reflect.ReflectModel): the model to calculate the gradient with.
        parameter (refnx.analysis.Parameter): the parameter to vary when calculating the gradient.
        q_point (float): the Q value of the model reflectivity point to calculate the gradient of.
        step (float): the step size to take when calculating the gradient.

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

def get_ground_truths(structure):
    """Gets the true values of the layers' thicknesses and SLDs of the given structure.

    Args:
        structure (refnx.reflect.Structure): the structure to get the true values from.

    """
    # Get the true values of the layers' thicknesses and SLDs.
    true = []
    for component in structure()[1:-1]:
        true.append(component.thick.value)
        true.append(component.sld.real.value)

    return np.asarray(true)

def usefulness(objective):
    """Calculate a usefulness metric for a given objective. This metric is
       intended to provide a measure of how closely the FIM results will
       match traditional sampling methods.

    Args:
        model (refnx.analysis.Objective): the objective to calculate the usefulness of.

    Returns:
        float: the usefulness metric for the objective.

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

def select_model(dataset, counts, layers=(1,5)):
    """Selects the best model for a given dataset.

    Args:
        dataset (refnx.dataset.ReflectDataset): the dataset to obtain a model for.
        counts (numpy.ndarray): neutron counts for each point in the dataset.
        layers (tuple): the range of layers to search through when selecting a model.
        
    Returns:
        refnx.reflect.ReflectModel: the selected model for the data.

    """
    generator = ModelGenerator()
    
    # Iterate over the layers to consider for the model.
    objectives, logls, AICs, BICs, KICs = [], [], [], [], []
    for layer in layers:
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
        g = calc_FIM(dataset.x, xi, counts, model)
        KICs.append(-2*logl - 2*logp + k*np.log(n/(2*np.pi)) + np.log(np.linalg.det(g/n)))
     
    # Display the best model using each information criterion.
    print('\nLog-likelihood: {}-layer'.format(np.argmax(logls)+1))  
    print('AIC: {}-layer'.format(np.argmin(AICs)+1))
    print('BIC: {}-layer'.format(np.argmin(BICs)+1))
    print('KIC: {}-layer'.format(np.argmin(KICs)+1))
    
    return objectives[np.argmin(AICs)] # Use the AIC by default.

if __name__ == '__main__':
    from refnx.analysis import Objective
    from simulate import simulate_single_contrast
    
    from structures import thin_layer_sample_1, thin_layer_sample_2
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import easy_sample, QCS_sample, many_param_sample
    from structures import STRUCTURES
    
    structure = easy_sample
    angle_times = {0.7: (70, 5),
                   2.0: (70, 20)}
    
    # Select a model for the simulated data of the chosen structure.
    _, dataset, counts = simulate_single_contrast(structure(), angle_times, include_counts=True)
    model = select_model(dataset, counts, layers=(1,4))

    # Calculate the usefulness of all structures in the structures file.
    print('----------- Usefulness Metrics -----------')
    for structure in STRUCTURES:
        objective = Objective(*simulate_single_contrast(structure(), angle_times))
        vary_structure(objective.model.structure, random_init=True)

        CurveFitter(objective).fit('differential_evolution', verbose=False)

        val = usefulness(objective)
        print('{0}: {1}'.format(structure.__name__, val))
