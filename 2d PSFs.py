# %%
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy import integrate

def PSF_lr(R):
    PSF = 1/(np.pi*(1+eta)) * (nu**(-2)/(1 + (np.abs(R)/gamma)**sigma) \
                               + eta/(beta**2) * np.exp(-(R**2)/beta**2))
    PSF_norm = PSF / integrate.simps(integrate.simps(PSF, x), y)
    return PSF_norm

def PSF_pl(R):
    PSF = 1/(np.pi*(1+eta)) * eta/(beta**2) * np.exp(-(R**2/beta**2))
    PSF_norm = PSF / integrate.simps(integrate.simps(PSF, x), y)
    return PSF_norm


# General parameters
gran = 100      # Granularity (nm)
pixel_size = gran/1000  # Convert to um

# Scattering parameters
beta = 6        # um, long range scattering
eta = 0.5       # dimensionless
gamma = 8.01    # nm
sigma = 2.60    # dimensionless
nu = 18.3       # nm

# There's definitely a better way to do this...
lb = - 3*round(beta)
ub = 3 * round(beta) + pixel_size
x = np.arange(lb, ub, pixel_size)
y = np.arange(lb, ub, pixel_size)
X, Y = np.meshgrid(x, y)
R = np.sqrt(np.abs(X)**2 + np.abs(Y)**2)
Z = np.array([PSF_lr(R), PSF_pl(R)])

fig, axs = plt.subplots(1, 2, figsize=(15,6), subplot_kw={"projection": "3d"})
for i, ax in enumerate(axs):
    surf = ax.plot_surface(X, Y, Z[i,:], cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter('{x:.02f}')
    # fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
# %%
