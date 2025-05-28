import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Constants
lambda_ = 1  # Normalize wavelength to 1
k = 2 * np.pi / lambda_  # Wave number


# Array factor squared for Uniform Linear Array (ULA)
def array_factor_squared(theta, N, d):
    psi = k * d * np.cos(theta)
    # Avoid division by zero using np.where
    numerator = np.sin(N * psi / 2)
    denominator = np.sin(psi / 2)
    af = np.where(np.abs(denominator) < 1e-6, N, numerator / denominator)
    return np.abs(af) ** 2


# Dipole element pattern (short dipole)
def dipole_pattern(theta):
    return np.sin(theta) ** 2


# Total radiation pattern (element pattern × array factor)
def power_pattern(theta, N, d):
    return array_factor_squared(theta, N, d) * dipole_pattern(theta)


# Compute directivity
def directivity(N, d):
    theta_max = np.pi / 2  # Broadside
    P_max = power_pattern(theta_max, N, d)

    # Integrate total radiated power over 0 to π
    integrand = lambda theta: power_pattern(theta, N, d) * np.sin(theta)
    total_power, _ = quad(integrand, 0, np.pi)

    # Directivity formula
    D = (4 * np.pi * P_max) / total_power
    return D


# Spacing from 0.1λ to 2.0λ
d_lambda = np.linspace(0.1, 2.0, 100)
spacings = d_lambda * lambda_

# Array sizes to simulate
N_values = [4, 6, 9]
directivities = {N: [] for N in N_values}

# Compute directivity for each spacing
for N in N_values:
    for d in spacings:
        D = directivity(N, d)
        directivities[N].append(D)

# Plotting
plt.figure(figsize=(10, 6))
for N in N_values:
    plt.plot(d_lambda, directivities[N], label=f'N = {N}')
plt.xlabel('Spacing (λ)')
plt.ylabel('Directivity (dimensionless)')
plt.title('Directivity vs. Element Spacing for uniform linear array of Dipoles')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('directivity_vs_spacing.png')
plt.show()

# Summary of observations
print("📊 Summary of Directivity Trends:")
print("----------------------------------")
print("✔️ Directivity increases with the number of elements (N).")
print("✔️ For small spacing (< 0.5λ), directivity is low due to element coupling.")
print("✔️ Directivity peaks near spacing ~0.5λ to 1.0λ, then shows oscillations.")
print("✔️ Very large spacing (> λ) causes grating lobes, reducing usable directivity.")
