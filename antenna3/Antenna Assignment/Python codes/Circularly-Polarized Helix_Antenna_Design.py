import numpy as np
import matplotlib.pyplot as plt

# Basic setup: define speed of light and operating frequency
c = 3e8                # speed of light in m/s
f = 600e6              # frequency of operation in Hz
wavelength = c / f     # compute wavelength from c = f * λ

# Parameters for the helix antenna in normal mode (shorter, more compact)
r_norm = 0.05 * wavelength      # radius is 5% of wavelength
S_norm = 0.1 * wavelength       # turn spacing is 10% of wavelength
N_norm = 4                      # total of 4 turns
C_norm = 2 * np.pi * r_norm     # circumference = 2πr

# Parameters for the helix antenna in axial mode (longer, directional)
r_axial = 0.16 * wavelength     # radius is 16% of wavelength
S_axial = 0.25 * wavelength     # spacing between turns is 25% of wavelength
N_axial = 8                     # total of 8 turns
C_axial = 2 * np.pi * r_axial   # compute circumference

# Function that generates the 3D coordinates of a helix based on radius, spacing, and number of turns
def generate_helix(r, S, N, num_points=1000):
    t = np.linspace(0, 2 * np.pi * N, num_points)  # angle range for the full helix
    x = r * np.cos(t)                             
    y = r * np.sin(t)                              
    z = S * t / (2 * np.pi)                        # z increases linearly for each full turn
    return x, y, z

# Generate coordinates for both normal and axial helix designs
x_norm, y_norm, z_norm = generate_helix(r_norm, S_norm, N_norm)
x_axial, y_axial, z_axial = generate_helix(r_axial, S_axial, N_axial)

# Plot the 3D geometry of both helix types for visual understanding
fig = plt.figure(figsize=(12, 5))

# 3D plot for normal mode
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot3D(x_norm, y_norm, z_norm, label='Normal Mode')
ax1.set_title("Normal Mode Helix")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.legend()

# 3D plot for axial mode
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot3D(x_axial, y_axial, z_axial, color='r', label='Axial Mode')
ax2.set_title("Axial Mode Helix")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
ax2.legend()

plt.tight_layout()
plt.show()

# Simulated radiation pattern for axial mode (focused beam)
# cos^2(theta/2) is a commonly used approximation for directional radiation
# theta from 0 to 2pi (360 degrees)
theta = np.linspace(0, 2*np.pi, 360)
gain_axial = 15 * (np.cos(theta / 2))**2  # gain in axial mode

# Radiation pattern for normal mode (omnidirectional pattern)
gain_normal = 5 * (np.sin(theta))**2      # gain in normal mode

# Creating a polar plots for both patterns side by side
plt.figure(figsize=(12, 6))

#Polar plot for normal mode (doughnut-shaped radiation)
plt.subplot(121, polar=True)
plt.plot(theta, gain_normal, color='green', label='Gain (sin²(θ))')
plt.title("Approximate Radiation Pattern (Normal Mode)", va='bottom')
plt.legend()

# Polar plot for axial mode (forward-directed beam)
plt.subplot(122, polar=True)
plt.plot(theta, gain_axial, color='blue', label='Gain (cos²(θ/2))')
plt.title("Approximate Radiation Pattern (Axial Mode)", va='bottom')
plt.legend()

plt.tight_layout()
plt.show()
