# Tools to validate input data. 

from astropy.table import Table


def validate_data(tpath, bands):
	if not tpath.endswith('fits'):
		raise InputError("Wide table must be in .fits format")

	t = Table.read(tpath, memmap=True)
	
	issues = []
	for b in list(bands):
		if 'Mf_%s'%b not in t.colnames:
			issues += ["Missing flux column Mf_%s"%b]
		if 'cov_Mf_%s'%b not in t.colnames:
			issues += ["Missing flux error column cov_Mf_%s"%b]

	if len(issues) > 0:
		print("Table Formatting Issues:")
		for isu in issues:
			print('\t- '+isu)
		print()

	return t

def validate_wide_data(tpath):
	t = validate_data(tpath, 'griz')

def validate_deep_data(tpath):
	
	t = validate_data(tpath, 'ugrizjhk')
	
	if 'COSMOS_PHOTZ' not in t.colnames:
		print("Deep Table missing 'COSMOS_PHOTZ' column")





