
'''
from PZC import *
from TomographicBins import *

pzc_path = 'fixtures/test_easy/pzc/'
pzc = load_PZC(pzc_path)
pzc.load_realizations()

tb_path = 'fixtures/test_easy/bins/'
tb = TomographicBins(pzc, tb_path)

res = tb.make_bins(4, compost_bin=0.05)

out = res.calculate_Nz(apply_selection_effects=True)
print(out[0].shape)
'''


from astropy.table import Table

t = Table.read('../data/deep_field_data/BFD/VALIDATION_cat_5E+04.fits')

print(t.colnames)





