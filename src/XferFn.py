
from abc import ABC, abstractmethod

class Simulations(ABC):

	@abstractmethod
	def generate_realizations(self):
		'''Defines how the realizations are made and stored. '''
		pass

	@abstractmethod
	def load_realizations(self, alternate_save_path=None):
		'''Once the realizations are made, defines how to retreive the generated realizations.'''
		pass



	


