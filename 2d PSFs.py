import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy import integrate, signal
import timeit



# -------- Parameters --------

# General parameters
gran = 100      # Granularity (nm)
pixel_size = gran/1000  # Convert to um

# Scattering parameters
beta = 6        # um, long range scattering
eta = 0.5       # dimensionless
gamma = 8.01    # nm
sigma = 2.60    # dimensionless
nu = 18.3       # nm



# -------- PSFs --------

# 2d coordinates for PSFs - from -beta to +beta (can increase but will slow down runtime)
# There's definitely a better way to do this...
lb = - 1 * round(beta)
ub = 1 * round(beta) + pixel_size
x = np.arange(lb, ub, pixel_size)
y = np.arange(lb, ub, pixel_size)
mesh = np.meshgrid(x, y)
r = np.sqrt(np.abs(mesh[0])**2 + np.abs(mesh[1])**2)

# Normalised PSFs
# Each array value corresponds to 100nm
PSF_lr = 1/(np.pi*(1+eta)) * eta/(beta**2) * np.exp(-np.abs(r)**2/beta**2)
PSF_lr = PSF_lr / integrate.simps(integrate.simps(PSF_lr, x), y)
PSF_pl = 1/(np.pi*(1+eta)) * (nu**(-2)/(1 + (np.abs(r)/gamma)**sigma) \
                           + eta/(beta**2) * np.exp(-np.abs((r)**2)/beta**2))
PSF_pl = PSF_pl / integrate.simps(integrate.simps(PSF_pl, x), y)
PSFs = np.array([PSF_lr, PSF_pl])



# -------- Properties of substrate --------

# 2d coordinates for substrate
subs_width = 40  # substrate width (um)
LB = 0
UB = subs_width + pixel_size
X = np.arange(LB, UB, pixel_size)
Y = np.arange(LB, UB, pixel_size)
Mesh = np.meshgrid(X, Y)

# Desired exposure pattern
D_0 = np.zeros((np.size(X), np.size(Y)))
startend = np.array([[[0, 10], [0, 40]],    # Start/end points of desired features (um)
                     [[11, 20], [0, 40]],
                     [[21, 22], [0, 40]],
                     [[30, 31], [0, 40]]])
for i, j in startend:
    D_0[int(i[0]/pixel_size) : int(i[1]/pixel_size), 0:-1] = 1



# %% Plots

# -------- Plot PSFs --------
fig, axs = plt.subplots(1, 2, figsize=(12,6), subplot_kw={"projection": "3d"})
for i, ax in enumerate(axs):
    surf = ax.plot_surface(mesh[0], mesh[1], PSFs[i,:], cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter('{x:.02f}')
    # fig.colorbar(surf, shrink=0.5, aspect=5)



# -------- Plot desired exposure dose --------
fig, ax = plt.subplots(figsize=(8,6), subplot_kw={"projection": "3d"})
surf = ax.plot_surface(Mesh[0], Mesh[1], D_0, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)



# -------- Plot corrected exposure dose --------
fig, ax = plt.subplots(figsize=(12,6), subplot_kw={"projection": "3d"})
fig.suptitle('Corrected exposure dose (NOT WORKING)')
axs[0].set_title('Long-range Gaussian only')
axs[1].set_title('Power law + long-range Gaussian')

# Long-range PSF
D_i = D_0
one = np.ones(np.shape(D_0))

for i in range(1):
    
    tic = timeit.default_timer()
    cnv = signal.convolve2d(D_i, PSF_lr, mode='same')
    toc = timeit.default_timer()
    print('Time to calculate convolution: ', toc-tic)
    
    cnv = signal.convolve2d(D_i, PSF_pl, mode='same')
    D_iplus1 = cnv # 2*D_0*(np.subtract(one,cnv))
    D_i = D_iplus1 # / np.max(D_iplus1)  # normalise
    
surf = ax.plot_surface(Mesh[0], Mesh[1], D_i, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Power law PSF
D_i = D_0
for i in range(2):
    tic = timeit.default_timer()
    cnv = signal.convolve2d(D_i, PSF_pl, mode='same')
    toc = timeit.default_timer()
    print('Time to calculate convolution: ', toc-tic)
    D_iplus1 = 2*D_0*(1-cnv) + cnv
    D_i = D_iplus1  / np.max(D_iplus1)  # normalise
surf = axs[1].plot_surface(Mesh[0], Mesh[1], D_i, cmap=cm.coolwarm, linewidth=0, antialiased=False)

plt.show()