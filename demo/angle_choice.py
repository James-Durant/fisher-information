import numpy as np

from refnx.reflect import SLD, Structure
from utils import first_angle_choice, second_angle_choice

def sample() -> Structure:
    """Defines a structure describing a sample with similar layer SLDs."""
    air = SLD(0, name='Air')
    layer1 = SLD(3.0, name='Layer 1')(thick=50, rough=2)
    layer2 = SLD(5.5, name='Layer 2')(thick=30, rough=6)
    layer3 = SLD(6.0, name='Layer 3')(thick=35, rough=2)
    substrate = SLD(2.047, name='Substrate')(thick=0, rough=2)
    return air | layer1 | layer2 | layer3 | substrate

# Path to directory to save results to.
save_path = './results'

points = 70 # Number of data points to simulate.
time = 1 # Time to use for simulation.

# Range of angles (in degrees) to calculate the FIM over.
angles = np.arange(0.3, 2.4, 0.05)

# Investigate how the FIM changes with first angle choice.
first_angle_choice(sample, angles, points, time, save_path)

# Investigate how the FIM changes with second angle choice.
initial_angle_times = {0.7: (70, 1)} # Angle: (Points, Time) for first angle.
second_angle_choice(sample, initial_angle_times, angles, points, time, save_path)
