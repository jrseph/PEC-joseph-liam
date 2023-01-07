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
    r = np.arange(-3*round(beta), 3*round(beta), 0.1)
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
    r = np.arange(-3*round(beta), 3*round(beta), 0.1)
    PSF = 1/(np.pi*(1+eta)) * (nu**(-2)/(1 + (np.abs(r)/gamma)**sigma) \
                               + eta/(beta**2) * np.exp(-np.abs((r)**2)/beta**2))
    PSF_norm = PSF / integrate.simps(PSF)
    return PSF_norm


# Scattering parameters
beta = 28.5     # um, long range scattering
eta = 0.75      # dimensionless
gamma = 8.01    # nm
sigma = 2.60    # dimensionless
nu = 18.3       # nm

# Porperties of substrate
subs_width = 200 # substrate width (um)
n_cells = subs_width * 1000   # number of r values such that each corresponds to a cell of width 100nm
x = np.linspace(-subs_width/2, subs_width/2, n_cells)


# Plot PSFs
plt.figure()
plt.xlabel(r'r [$\mu \mathrm{m}$]')
plt.ylabel(r'PSF [$\mu \mathrm{m} ^{-2}$]')
plt.title('Normlised point-spread functions')
plt.plot(PSF_pl(), label='power law + Gaussian')
plt.plot(PSF_lr(), label='long range Gaussian only')
plt.legend()

'''
# Maps out dose pattern WITHOUT scattering for a number of pointlike features at r=r_f
# Dose at each r value is either 1 or 0
r_f = np.array([10, 20, 30, 40, 80, 100, 150, 190])       # the r-position of the features (um)
beam_rad = 5    # beam radius (nm)
D_i = np.zeros(n_cells)
for i in r_f:
    D_i[int(np.round(n_cells * i/subs_width)-beam_rad) : int(np.round(n_cells * i/subs_width)+beam_rad)] = 1

plt.figure()
plt.plot(D_i)
plt.title('Dose pattern without electron scattering')

plt.figure()
plt.plot(np.convolve(D_i, PSF_lr(r, beta=5, eta=eta)[1])) # , label='$\beta$ = {}'.format(b))
'''
plt.show()