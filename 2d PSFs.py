import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy import integrate

def PSF_lr(r):
    PSF = 1/(np.pi*(1+eta)) * (nu**(-2)/(1 + (np.abs(r)/gamma)**sigma) \
                               + eta/(beta**2) * np.exp(-(r**2)/beta**2))
    PSF_norm = PSF / integrate.simps(integrate.simps(PSF, x), y)
    return PSF_norm

def PSF_pl(r):
    PSF = 1/(np.pi*(1+eta)) * eta/(beta**2) * np.exp(-(r**2/beta**2))
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

# -------- 2d coordinates for PSFs --------
# There's definitely a better way to do this...
lb = - 3*round(beta)
ub = 3 * round(beta) + pixel_size
x = np.arange(lb, ub, pixel_size)
y = np.arange(lb, ub, pixel_size)
mesh = np.meshgrid(x, y)
r = np.sqrt(np.abs(mesh[0])**2 + np.abs(mesh[1])**2)
PSFs = np.array([PSF_lr(r), PSF_pl(r)])

# -------- 2d coordinates for substrate --------
subs_width = 40  # substrate width (um)
LB = 0
UB = subs_width + pixel_size
X = np.arange(LB, UB, pixel_size)
Y = np.arange(LB, UB, pixel_size)
Mesh = np.meshgrid(X, Y)

# Map out desired feature pattern WITHOUT considering scattering
# Dose at each r value is either 1 or 0
D_0 = np.zeros((np.size(X), np.size(Y)))
# Start/end X/Y points of desired features (um)
startend = np.array([[[0, 10], [0, 40]],
                     [[11, 20], [0, 40]],
                     [[21, 22], [0, 40]],
                     [[30, 31], [0, 40]]])
for i, j in startend:
    D_0[int(i[0]/pixel_size) : int(i[1]/pixel_size), int(j[0]/pixel_size) : int(j[1]/pixel_size)] = 1
#     D_0[int(i[0]/cell_size) : int(i[1]/cell_size)] = 1

print(np.shape(PSFs[0]))
print(np.shape(PSFs[0, :]))
print(np.shape(D_0))

fig, axs = plt.subplots(1, 2, figsize=(15,6), subplot_kw={"projection": "3d"})
for i, ax in enumerate(axs):
    surf = ax.plot_surface(mesh[0], mesh[1], PSFs[i,:], cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter('{x:.02f}')
    # fig.colorbar(surf, shrink=0.5, aspect=5)

fig, ax = plt.subplots(figsize=(15,6), subplot_kw={"projection": "3d"})
surf = ax.plot_surface(Mesh[0], Mesh[1], D_0, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

plt.show()