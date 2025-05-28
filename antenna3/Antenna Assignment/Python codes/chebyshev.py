import numpy as np
import matplotlib.pyplot as plt
from scipy.special import chebyt

# Constants
lambda_ = 1  # Wavelength (normalized)
k = 2 * np.pi / lambda_  # Wave number
d = 0.5 * lambda_  # Spacing (0.5 lambda)
N = 10  # Number of elements
sll_dB = [20, 30, 40]  # Chebyshev side-lobe levels in dB
tapers = ['Uniform'] + [f'Chebyshev {sll} dB' for sll in sll_dB]  # Tapers

# Dipole element pattern (power pattern)
def dipole_pattern(theta):
    return np.sin(theta)**2

# Chebyshev weights for a given SLL and N
def chebyshev_weights(N, sll_dB):
    R = 10 ** (sll_dB / 20)  # SLL ratio
    x0 = np.cosh(np.arccosh(R) / (N - 1))  # Chebyshev parameter
    T = chebyt(N - 1)  # Chebyshev polynomial of order N-1
    weights = []
    for m in range(N):
        sum_term = 0
        for p in range(N):
            xp = np.cos(np.pi * (p + 0.5) / N)
            Tp = T(x0 * xp)
            sum_term += Tp * np.cos(np.pi * m * (p + 0.5) / N)
        weight = (1/N) * (1 + 2 * sum_term * (R - 1) / (T(x0) - 1))
        weights.append(weight)
    return np.array(weights)

# Array factor for given weights
def array_factor(theta, N, weights):
    psi = k * d * np.cos(theta)
    af = 0
    for n in range(N):
        af += weights[n] * np.exp(1j * n * psi)
    return np.abs(af)**2 * dipole_pattern(theta)

# Find 3 dB beamwidth and SLL
def compute_beamwidth_sll(theta, af):
    af_max = np.max(af)
    af_dB = 10 * np.log10(af / af_max + 1e-10)

    # Find 3 dB points (where power drops to 1/sqrt(2))
    idx_90 = np.argmin(np.abs(theta - np.pi/2))  # Index at theta = 90 deg
    af_3dB = af_max / np.sqrt(2)
    idx_left = np.where(af[:idx_90] <= af_3dB)[0]
    idx_right = np.where(af[idx_90:] <= af_3dB)[0]
    if len(idx_left) == 0 or len(idx_right) == 0:
        beamwidth = np.nan
    else:
        theta_left = theta[idx_left[-1]]
        theta_right = theta[idx_90 + idx_right[0]]
        beamwidth = np.degrees(theta_right - theta_left)

    # Find SLL (maximum side-lobe peak outside main lobe)
    main_lobe_region = (theta > np.pi/2 - 0.2) & (theta < np.pi/2 + 0.2)  # Approx main lobe
    side_lobes = af_dB[~main_lobe_region]
    sll = np.max(side_lobes) if side_lobes.size > 0 else np.nan

    return beamwidth, sll

# Compute and plot array factor for each Chebyshev taper vs Uniform
theta = np.linspace(0, np.pi, 1000)  # Angle from 0 to 180 degrees
colors = ['k', 'b', 'g', 'r']  # Add enough colors

results = {
    sll: {
        'Uniform': {'beamwidth': None, 'sll': None},
        f'Chebyshev {sll} dB': {'beamwidth': None, 'sll': None}
    }
    for sll in sll_dB
}

for i, sll in enumerate(sll_dB):
    plt.figure(figsize=(8, 6))

    # Uniform taper
    weights_uniform = np.ones(N)
    af_uniform = np.array([array_factor(t, N, weights_uniform) for t in theta])
    af_uniform_dB = 10 * np.log10(af_uniform / np.max(af_uniform) + 1e-10)
    bw_uniform, sll_uniform = compute_beamwidth_sll(theta, af_uniform)
    results[sll]['Uniform']['beamwidth'] = bw_uniform
    results[sll]['Uniform']['sll'] = sll_uniform
    plt.plot(np.degrees(theta), af_uniform_dB,
             label=f'Uniform: BW={bw_uniform:.2f}°, SLL={sll_uniform:.2f} dB',
             color=colors[0])

    # Chebyshev taper
    weights_cheby = chebyshev_weights(N, sll)
    af_cheby = np.array([array_factor(t, N, weights_cheby) for t in theta])
    af_cheby_dB = 10 * np.log10(af_cheby / np.max(af_cheby) + 1e-10)
    bw_cheby, sll_cheby = compute_beamwidth_sll(theta, af_cheby)
    results[sll][f'Chebyshev {sll} dB']['beamwidth'] = bw_cheby
    results[sll][f'Chebyshev {sll} dB']['sll'] = sll_cheby
    plt.plot(np.degrees(theta), af_cheby_dB,
             label=f'Chebyshev {sll} dB: BW={bw_cheby:.2f}°, SLL={sll_cheby:.2f} dB',
             color=colors[1])

    # Plot setup
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Array Factor (dB)')
    plt.title(f'Array Factor Comparison: Uniform vs Chebyshev {sll} dB, N = {N}, d = 0.5λ')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.ylim(-60, 0)
    plt.tight_layout()
    plt.savefig(f'array_factor_n10_uniform_vs_chebyshev_{sll}dB.png')
    plt.close()
