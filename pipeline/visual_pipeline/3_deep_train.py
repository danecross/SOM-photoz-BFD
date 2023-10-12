
from astropy.table import Table

from NoiseSOM import *


fname = '../data/deep_field_data/BFD/TRAIN_CAT_5E+04.fits'
t = Table.read(fname, format='fits')

fluxes_colname = 'DESD_flux_ugrizjhk'
fluxes_err_colname = 'DESD_flux_ugrizjhk_err'

fluxes = t[fluxes_colname].data
fluxes_err = t[fluxes_err_colname].data

CAT_TYPE = 'BFD'
out_path = '../outputs/%s_deep/'%CAT_TYPE

deepres = 64

nTrain=fluxes.shape[0]
hh = hFunc(nTrain,sigma=(30,1))
metric = AsinhMetric(lnScaleSigma=0.4,lnScaleStep=0.03)
indices = np.random.choice(fluxes.shape[0],size=nTrain,replace=False)
som = NoiseSOM(metric,fluxes[indices,:],fluxes_err[indices,:], \
                            learning=hh, \
                            shape=(deepres,deepres), \
                            wrap=False,logF=True, \
                            initialize='sample', \
                            minError=0.02)

som.train()
som.save(path=out_path)




