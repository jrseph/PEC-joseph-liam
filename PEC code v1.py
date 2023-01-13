import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, signal
import timeit


# -------- Parameters --------

# General parameters
gran = 100  # Granularity (nm)
cell_size = gran/1000  # Convert to um

# Scattering parameters
beta = 6     # um, long range scattering
eta = 0.5      # dimensionless
gamma = 8.01    # nm
sigma = 2.60    # dimensionless
nu = 18.3       # nm



# -------- PSFs --------

# Coordinates for PSFs - from -2beta to +2beta
r = np.arange(-3*round(beta), 3*round(beta)+cell_size, cell_size)

# Normalised PSFs
# Each array value corresponds to 100nm
PSF_lr = 1/(np.pi*(1+eta)) * eta/(beta**2) * np.exp(-np.abs(r)**2/beta**2)
PSF_lr = PSF_lr / integrate.simps(PSF_lr)
PSF_pl = 1/(np.pi*(1+eta)) * (nu**(-2)/(1 + (np.abs(r)/gamma)**sigma) \
                           + eta/(beta**2) * np.exp(-np.abs((r)**2)/beta**2))
PSF_pl = PSF_pl / integrate.simps(PSF_pl)
PSFs = np.array([PSF_lr, PSF_pl])



# -------- Properties of substrate --------

# Substrate coordinates
subs_width = 40  # substrate width (um)
x = np.arange(0, subs_width+cell_size, cell_size)

# Desired exposure pattern
# Dose at each r value is either 1 or 0
D_0 = np.zeros(np.size(x))
startend = np.array([[0, 10],   # Start/end points of desired features (um)
                     [11, 20],
                     [21, 22],
                     [30, 31]])
for i in startend:
    D_0[int(i[0]/cell_size) : int(i[1]/cell_size)] = 1



# -------- Plot PSFs --------
# fig, ax = plt.subplots()
# ax.set_xlabel(r'r [$\mu \mathrm{m}$]')
# ax.set_ylabel(r'PSF [$\mu \mathrm{m} ^{-2}$]')
# ax.set_title('Normlised point-spread functions')
# ax.plot(PSF_pl, label='power law + Gaussian')
# ax.plot(PSF_lr, label='long range Gaussian only')
# ax.legend()



# # -------- Plot desired exposure dose --------
# fig, ax = plt.subplots()
# ax.set_title('Desired dose pattern as in Watson paper')
# ax.plot(x, D_0)



# -------- Plot corrected exposure doses --------
fig, axs = plt.subplots(1,2, figsize=(12, 6))
fig.suptitle('Corrected exposure dose')
axs[0].set_title('Long-range Gaussian only')
axs[1].set_title('Power law + long-range Gaussian')

tic = timeit.default_timer()

for n, fn in enumerate(PSFs):
    D_i = D_0
    for i in range(4):
        cnv = signal.convolve(D_i, fn, mode='same')
        D_iplus1 = 2*D_0*(1-cnv) # + cnv
        D_i = D_iplus1 / np.max(D_iplus1)  # normalise
        
    axs[n].plot(x, D_i)

toc = timeit.default_timer()
print('Time to calculate corrected exposure doses: ', toc-tic)

plt.show()