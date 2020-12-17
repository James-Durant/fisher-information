import numpy as np

def calc_FIM(q, xi, flux, model):
    """Calculates the Fisher information matrix for given a dataset and model.

    Args:
        q (numpy.ndarray): array of Q values.
        xi (list): list of refnx Parameter objects representing each varying parameter value.
        flux (numpy.ndarray): array of flux values corresponding to each Q value.
        model (refnx.reflect.ReflectModel): the model to calculate the gradient with.

    Returns:
        numpy.ndarray: Fisher information matrix for the model and data.

    """
    n = len(q)
    m = len(xi)
    J = np.zeros((n,m))
    #Calculate the gradient of the model reflectivity with every model parameter for every model data point.
    for i in range(n):
        for j in range(m):
            J[i,j] = gradient(model, xi[j], q[i])

    r = model(q) #Use model reflectivity values
    M = np.diag(flux/r, k=0)
    g = np.dot(np.dot(J.T, M), J)
    return g

def gradient(model, parameter, q_point, step=0.005):
    """Calculate a two-point gradient of model reflectivity with model `parameter`.

    Args:
        model (refnx.reflect.ReflectModel): the model to calculate the gradient with.
        parameter (refnx.analysis.Parameter): the parameter to vary when calculating the gradient.
        q_point (float): the Q value of the model reflectivity point to calculate the gradient of.
        step (float): the step size to take when calculating the gradient.

    Returns:
        float: two-point gradient.

    """
    step = parameter.value * step #0.5% step by default
    old = parameter.value

    x1 = parameter.value = old - step #First point
    y1 = model(q_point) #Get new r value with altered model.

    x2 = parameter.value = old + step #Second point
    y2 = model(q_point)

    parameter.value = old #Reset parameter
    return (y2-y1) / (x2-x1) #Return the gradient
