import numpy as np
import sys
sys.path.append('../fisher-information')

from refnx.reflect import SLD, Structure
from experimental_design import first_angle_choice, second_angle_choice

def similar_sld_sample() -> Structure:
    """Defines a structure describing a sample with similar layer SLDs."""
    air = SLD(0, name='Air')
    layer1 = SLD(3.0, name='Layer 1')(thick=50, rough=2)
    layer2 = SLD(5.5, name='Layer 2')(thick=30, rough=6)
    layer3 = SLD(6.0, name='Layer 3')(thick=35, rough=2)
    substrate = SLD(2.047, name='Substrate')(thick=0, rough=2)
    return air | layer1 | layer2 | layer3 | substrate

points = 70 # Number of data points to simulate.
time = 1 # Time to use for simulation.

# Angles (in degrees) to calculate the FI over.
angles = np.arange(0.25, 2.5, 0.01)

# Investigate how the FI changes with first angle choice.
first_angle_choice(similar_sld_sample, angles, points, time, './results')

# Investigate how the FI changes with second angle choice.
initial_angle_times = {0.7: (70, 1)} # Angle: (Points, Time) for first angle.
second_angle_choice(similar_sld_sample, initial_angle_times, angles, points, time, './results')
