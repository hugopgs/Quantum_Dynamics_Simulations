import numpy as np
import matplotlib.pyplot as plt

def Pe_ramsey_sequencies(delta, Omega, tau, T):
    """Transition probability P_e(Δ) for a Ramsey sequence (exact formula)."""
    Omega_eff = np.sqrt(Omega**2 + delta**2)
    c = np.cos(Omega_eff * tau / 2)
    s = np.sin(Omega_eff * tau / 2)
    alpha = delta / Omega_eff
    beta = Omega / Omega_eff
    term = c * np.cos(delta * T / 2) + alpha * s * np.sin(delta * T / 2)
    return 4 * beta**2 * s**2 * term**2




def plot_ramsey(Omega=1.0):
    delta = np.linspace(-10 * Omega, 10 * Omega, 2000)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    # ---- (1) Compare different T values for π/2 pulse ----
    tau = np.pi / (2 * Omega)  # π/2 pulse
    for i in range(1, 3):
        T = i * (5 / Omega)
        Pe_values = Pe_ramsey_sequencies(delta, Omega, tau, T)
        axes[0].plot(delta / Omega, Pe_values, lw=2, label=fr'$T={i*5}/\Omega$')
    axes[0].legend()
    axes[0].set_ylabel(r'$P_e(\Delta)$', fontsize=12)
    axes[0].set_title(r'Ramsey fringes for $\tau\Omega=\pi/2$ and varying $T$')
    axes[0].grid(True, ls=':')

    # ---- (2) Compare π/2 vs π pulses for fixed T ----
    T = 10 / Omega
    for tau_label, tau_val in [("π/2", np.pi / (2 * Omega)), ("π", np.pi / Omega)]:
        Pe_values = Pe_ramsey_sequencies(delta, Omega, tau_val, T)
        axes[1].plot(delta / Omega, Pe_values, lw=2, label=fr'$\tau\Omega={tau_label}$')
    axes[1].legend()
    axes[1].set_xlabel(r'$\Delta / \Omega$', fontsize=12)
    axes[1].set_ylabel(r'$P_e(\Delta)$', fontsize=12)
    axes[1].set_title(r'Effect of pulse area for $T=10/\Omega$')
    axes[1].grid(True, ls=':')

    plt.tight_layout()
    plt.show()


# Run the function
plot_ramsey()
