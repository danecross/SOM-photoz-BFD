
import pytest

from SOM import *
from PZC import *
from TomographicBins import *

wideSOM_path = 'fixtures/test_easy/wide_fluxes.fits'
deepSOM_path = 'fixtures/test_easy/deep_fluxes.fits'

sky_cov_path = 'fixtures/test_easy/covariances.pkl'

wideres = 10 ; deepres = 15

wide_output = 'fixtures/test_easy/wide/'
deep_output = 'fixtures/test_easy/deep/'
pzc_output = 'fixtures/test_easy/pzc/'
tb_output = 'fixtures/test_easy/bins/'

class TestEasy:

	##################
	## SOM training ##
	##################

	@pytest.fixture(scope="session")
	def wideSOM(self, tmpdir_factory):
		if not os.path.exists(wide_output): os.mkdir(wide_output)
		wide_SOM = SOM(wideres, wideSOM_path, wideSOM_path, analysis_output_path=wide_output)
		wide_SOM.train()
		wide_SOM.validate()
		wide_SOM.save(wide_output+'SOM.pkl')

		return wide_SOM

	@pytest.fixture(scope="session")
	def deepSOM(self, tmpdir_factory):
		if not os.path.exists(deep_output): os.mkdir(deep_output)
		deep_SOM = SOM(deepres, deepSOM_path, deepSOM_path, analysis_output_path=deep_output)
		deep_SOM.train()
		deep_SOM.validate()
		deep_SOM.save(deep_output+'SOM.pkl')

		return deep_SOM

	def test_SOM_training(self, wideSOM, deepSOM):
		assert(wideSOM.trained)
		assert(deepSOM.trained)

	#######################
	## P(z|chat) mapping ##
	#######################

	@pytest.fixture(scope="session")
	def pzc_init(self, tmpdir_factory, wideSOM, deepSOM):
		if not os.path.exists(pzc_output): os.mkdir(pzc_output)
		pzc = PZC(wideSOM, deepSOM, pzc_output, sky_cov_path)
		return pzc

	@pytest.fixture(scope="session")
	def pzc_generated(self, tmpdir_factory, wideSOM, deepSOM):
		pzc = PZC(wideSOM, deepSOM, pzc_output, sky_cov_path)
		pzc.generate_realizations()
		return pzc

	@pytest.fixture(scope="session")
	def pzc(self, tmpdir_factory, pzc_generated):
		pzc_generated.make_redshift_map()
		pzc_generated.save(pzc_output)
		return pzc_generated


	def test_PZC_init(self, pzc_init):
		assert(hasattr(pzc_init, 'n_realizations'))

	def test_PZC_generation(self, pzc_generated):
		assert(os.path.exists(os.path.join(pzc_output, "simulations.fits")))

	def test_PZC(self, pzc):
		assert(hasattr(pzc, 'pzchat'))


	#########################
	## Tomographic Binning ##
	#########################

	@pytest.fixture(scope="session")
	def tb(self, pzc):
		if not os.path.exists(tb_output): os.mkdir(tb_output)
		tb = TomographicBins(pzc, tb_output, large_data_path=wideSOM_path)
		tb.classify()

		return tb

	def test_tomographic_binning(self, tb):
		bins = tb.make_bins(4)
		bins.save(tb_output+"Nz.pkl")






