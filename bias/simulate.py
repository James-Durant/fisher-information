import numpy as np

from refnx.dataset import ReflectDataset
from refnx.reflect import ReflectModel

def simulate(structure, angle_times, dq=2, bkg_rate=1e-7, include_counts=False):
    #Define the model. 0 background for simulating as background noise is added.
    model = ReflectModel(structure, scale=1, dq=dq, bkg=0)

    q, r, r_error, counts = [], [], [], []
    total_points = 0
    for angle in angle_times:
        #Simulate the experiment for the angle.
        points, time  = angle_times[angle]
        total_points += points
        q_angle, r_angle, r_error_angle, counts_angle = run_experiment(model, angle, points, time, bkg_rate)

        #Combine the data for the angle with the data from other angles.
        q       += q_angle
        r       += r_angle
        r_error += r_error_angle
        counts  += counts_angle

    #If fitting, the model needs to have the right background.
    model.bkg.value = bkg_rate

    if include_counts:
        data = np.zeros((total_points, 4))
        data[:,0] = q
        data[:,1] = r
        data[:,2] = r_error
        data[:,3] = counts
    else:
        data = np.zeros((total_points, 3))
        data[:,0] = q
        data[:,1] = r
        data[:,2] = r_error

    data = data[(data != 0).all(1)]  #Remove points of zero reflectivity.
    data = data[data[:,0].argsort()] #Sort by Q

    if include_counts:
        return model, data
    else:
        return model, ReflectDataset([data[:,0], data[:,1], data[:,2]])

def run_experiment(model, angle, points, time, bkg_rate=1e-7, 
                   directbeam_file="./data/directbeam_wavelength.dat"):
    #Load the directbeam_wavelength.dat file.
    direct_beam = np.loadtxt(directbeam_file, delimiter=',')
    wavelengths = direct_beam[:,0] # 1st column is wavelength, 2nd is flux.
    #Adjust flux by measurement angle.
    direct_flux = direct_beam[:,1] * pow(angle/0.3, 2)

    theta = angle*np.pi / 180 #Convert angle from degrees into radians.
    q = 4*np.pi*np.sin(theta) / wavelengths #Calculate Q values.

    #Bin Q values in equally log-spaced bins using flux as weighting.
    q_bin_edges = np.logspace(np.log10(np.min(q)), np.log10(np.max(q)), points+1)

    flux_binned, _ = np.histogram(q, q_bin_edges, weights=direct_flux)

    #Get the bin centres and calculate model reflectivity.
    q_binned = [(q_bin_edges[i] + q_bin_edges[i+1]) / 2 for i in range(points)]
    reflectance = model(q_binned)

    r, r_errors, counts = [], [], []
    for i in range(points): #Iterate over the desired number of points (bins).
        flux_point = flux_binned[i]
        r_point    = reflectance[i]
        count_incident = flux_point * time

        #Add background noise and get the measured reflected count for the bin.
        count_reflected = np.random.poisson((r_point*flux_point + bkg_rate) * time)

        #Point has zero reflectivity if there is no flux.
        if count_reflected > 0:
            #Convert from count space to reflectivity space.
            r_noisy = count_reflected / count_incident
            r_error = np.sqrt(count_reflected) / count_incident

            r.append(r_noisy)
            r_errors.append(r_error)
            counts.append(count_incident) #Incident neutrons in the bin
        else:
            r.append(0)
            r_errors.append(0)
            counts.append(0) #Incident neutrons in the bin

    return q_binned, r, r_errors, counts
