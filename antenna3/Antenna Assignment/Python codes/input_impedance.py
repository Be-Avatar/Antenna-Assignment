import numpy as np
import matplotlib.pyplot as plt
def dipole_impedance_approx(length):
    """
    Approximate impedance for center-fed dipoles (empirical fit).
    Valid for 0.1λ < L < 2.5λ.
    """
    # Approximate from standard curve: Z0 = R + jX
    R = 73 * np.sin(np.pi * length / 2)**2 / (np.pi * length / 2)**2
    X = 42.5 * np.log10(length / 0.5)  # crude approximation
    return R + 1j * X

lengths = np.linspace(0.1, 2.5, 100)
impedances = [dipole_impedance_approx(l) for l in lengths]

r = [z.real for z in impedances]
x = [z.imag for z in impedances]

plt.figure(figsize=(10, 5))
plt.plot(lengths, r, label='Resistance (Approx)')
plt.plot(lengths, x, label='Reactance (Approx)')
plt.xlabel('Dipole Length (λ)')
plt.ylabel('Impedance (Ohms)')
plt.title('Dipole Input Impedance vs Length (Approximation)')
plt.grid(True)
plt.legend()
plt.show()

