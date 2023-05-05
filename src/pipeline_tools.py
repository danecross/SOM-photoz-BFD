
import numpy as np

# converts flux to magnitude
def flux_to_mag(f, const=30):
    return -2.5 * np.log10(f) + const

# calculates total signal to noise
def SN(fluxes, fluxes_err):
    g = fluxes[:,0] ; r = fluxes[:,1]
    i = fluxes[:,2] ; z = fluxes[:,3]
    
    ge = fluxes_err[:,0] ; re = fluxes_err[:,1]
    ie = fluxes_err[:,2] ; ze = fluxes_err[:,3]
    
    signal = np.sqrt(0.7 * r**2 + 0.2 * i**2 + 0.1 * z**2)
    noise = np.sqrt(0.7 * re**2 + 0.2 * ie**2 + 0.1 * ze**2)
    return signal/noise

