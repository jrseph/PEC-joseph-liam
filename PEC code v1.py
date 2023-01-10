import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def PSF_lr():
    '''
    Normalised point-spread function as a long-range Gaussian only
    Each array value corresponds to 100nm
    
    beta : long-range scattering characteristic length (um)
    eta : Gaussian weighting (dimensionless)
    '''
    # PSF runs from -3*beta to 3*beta; each r val corresponds to 100nm if beta is given in um
    r = np.arange(-3*round(beta), 3*round(beta)+cell_size, cell_size)
    PSF = 1/(np.pi*(1+eta)) * eta/(beta**2) * np.exp(-np.abs(r**2)/beta**2)
    PSF_norm = PSF / integrate.simps(PSF)
    return PSF_norm


def PSF_pl():
    '''
    Normalised oint-spread function as a (short-range) power law plus a long-range Gaussian
    Each array value corresponds to 100nm
    
    beta : long-range scattering characteristic length (um)
    eta : Gaussian weighting (dimensionless)
    sigma : power-law scaling of intermdiate range (dimensionless)
    gamma : short range over which dose distribution is mostly constant (nm)
    nu : chosen to normalise PSF (nm)
    '''
    # PSF runs from -3*beta to 3*beta; each r val corresponds to 100nm if beta is given in um
    r = np.arange(-3*round(beta), 3*round(beta)+cell_size, cell_size)
    PSF = 1/(np.pi*(1+eta)) * (nu**(-2)/(1 + (np.abs(r)/gamma)**sigma) \
                               + eta/(beta**2) * np.exp(-np.abs((r)**2)/beta**2))
    PSF_norm = PSF / integrate.simps(PSF)
    return PSF_norm


def cnv_tr(dose, PSF):
    cnv = np.convolve(dose, PSF)
    cnv_tr = cnv[int(np.round(np.size(PSF)/2)):-int(np.round(np.size(PSF)/2))]
    return cnv_tr


# General parameters
gran = 100  # Granularity (nm)
cell_size = gran/1000  # Convert to um

# Scattering parameters
beta = 6     # um, long range scattering
eta = 0.5      # dimensionless
gamma = 8.01    # nm
sigma = 2.60    # dimensionless
nu = 18.3       # nm

# Porperties of substrate
subs_width = 40  # substrate width (um)
x = np.arange(0, subs_width+cell_size, cell_size)
n_cells = np.size(x)
# Map out desired feature pattern WITHOUT considering scattering
# Dose at each r value is either 1 or 0
D_0 = np.zeros(np.size(x))
# Start/end points of desired features (um)
startend = np.array([[0, 10],
                     [11, 20],
                     [21, 22],
                     [30, 31]])
for i in startend:
    D_0[int(i[0]/cell_size) : int(i[1]/cell_size)] = 1


# %% Plots

# Plot PSFs
fig, ax = plt.subplots()
ax.set_xlabel(r'r [$\mu \mathrm{m}$]')
ax.set_ylabel(r'PSF [$\mu \mathrm{m} ^{-2}$]')
ax.set_title('Normlised point-spread functions')
ax.plot(PSF_pl(), label='power law + Gaussian')
ax.plot(PSF_lr(), label='long range Gaussian only')
ax.legend()

# Plot desired features
fig, ax = plt.subplots()
ax.set_title('Desired dose pattern as in Watson paper')
ax.plot(x, D_0)

# Plot convolution
fig, ax = plt.subplots()
'''
D_i = D_0
for i in range(20):
    D_iplus1 = 2 * D_0 * (1 - cnv_tr(D_i, PSF_lr()))
    D_i = D_iplus1 / np.max(D_iplus1)  # normalise
# ax.plot(x, D_i)
'''
D_i = D_0
for i in range(20):
    D_iplus1 = 2 * D_0 * (1 - cnv_tr(D_i, PSF_lr())) + cnv_tr(D_i, PSF_lr())
    D_i = D_iplus1  / np.max(D_iplus1)  # normalise
ax.plot(x, D_i)

plt.show()