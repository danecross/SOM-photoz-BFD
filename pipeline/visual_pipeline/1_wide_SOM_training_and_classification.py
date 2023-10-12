#!/usr/bin/env python
# coding: utf-8

# In[1]:


import multiprocessing as mp
import tqdm
import numpy as np
from astropy.table import Table
import pickle

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# # SOM Training and Classification
# 
# This project uses the SOM training and classficiation algorthms described in [Sánchez et. al. (2020)](https://arxiv.org/pdf/2004.09542.pdf). 

# In[2]:


from NoiseSOM import *
from pipeline_tools import *


# ## Training

# In[3]:


fname = '../data/wide_field_data/TRAIN_CAT_1E+05.fits'
t = Table.read(fname, format='fits')


# In[4]:


fluxes = np.array([t['Mf_%s'%s] for s in ['g', 'r', 'i', 'z']]).T
fluxes_cov = np.array([t['cov_Mf_%s'%s] for s in ['g', 'r', 'i', 'z']]).T
fluxes_err = np.sqrt(fluxes_cov)


# In[5]:


CAT_TYPE = 'BFD'
out_path = '../outputs/%s/'%CAT_TYPE

somres = 32


# In[6]:


if not os.path.exists(out_path+'SOM.pkl'):
    nTrain=fluxes.shape[0]
    hh = hFunc(nTrain,sigma=(30,1))
    metric = AsinhMetric(lnScaleSigma=0.4,lnScaleStep=0.03)
    indices = np.random.choice(fluxes.shape[0],size=nTrain,replace=False)
    som = NoiseSOM(metric,fluxes[indices,:],fluxes_err[indices,:], \
                                learning=hh, \
                                shape=(somres,somres), \
                                wrap=False,logF=True, \
                                initialize='sample', \
                                minError=0.02)

    som.train()
    som.save(path=out_path)
else:
    print("loading SOM from %sSOM.pkl"%out_path)
    som = load_SOM(out_path+'SOM.pkl')


# ## Classification 
# 
# Now that the SOM is trained, we will use the validation catalog to ensure that the SOM algorithm worked correctly.

# In[7]:


fname = '../data/wide_field_data/VALIDATION_CAT_1E+06.fits'
t = Table.read(fname, format='fits')

fluxes = np.array([t['Mf_%s'%s] for s in ['g', 'r', 'i', 'z']]).T
fluxes_cov = np.array([t['cov_Mf_%s'%s] for s in ['g', 'r', 'i', 'z']]).T
fluxes_err = np.sqrt(fluxes_cov)


# ### Assigning Galaxies to Cells
# 
# Now we're going to run the Noise_SOM `classify` function to classify the random subset of the catalog. This can be done in parallel, so let's define a function that can be run in multiprocessing:

# In[8]:


num_inds = 10000
inds = np.array_split(np.arange(len(t)),num_inds)
def assign_som(index):
    cells_test, _ = som.classify(fluxes[inds[index]], 
                                     fluxes_err[inds[index]])
    
    return cells_test


# In[10]:


# Now running the multiprocessing pool (this can take a few minutes)
def run_pools():
    
    filename = "%s/assignments.pkl"%out_path
    if True: #not os.path.exists(filename):
        print("Assigning Galaxies...")
        with mp.Pool(4) as p: 
            args = [(i,) for i in range(num_inds)]
            results = list(tqdm.tqdm(p.imap(assign_som, range(num_inds)), total=num_inds))
        
        assignments = []
        for res in results:
            assignments = np.append(assignments,res)
            
        with open(filename, 'wb') as f:
            pickle.dump(assignments, f)
            f.close()
    else:
        print("Loading Galaxy Assignments...")
        with open(filename, 'rb') as f:
            assignments = pickle.load(f)
        
    print("Done")
    return assignments
    
    
if __name__ == '__main__':
    assignments = run_pools()
    
    t['Wide Cell Assignment'] = assignments


# ### Calculating aggregate properties of the galaxies in each cell
# 
# Here we use the [astropy aggregate function](https://docs.astropy.org/en/stable/api/astropy.table.TableGroups.html#astropy.table.TableGroups.aggregate) to group the table based on the wide cell assignments and perform aggregate functions on them, in this case average and standard deviation. 

# In[ ]:


mags = flux_to_mag(fluxes)
bands = ['g', 'r', 'i', 'z']
for i,b in enumerate(bands):
    t['%s_mag'%b] = mags[:,i]
    t['SN_%s'%b] = (fluxes/np.sqrt(fluxes_cov))[:,i]
    
colors = ['g-r', 'r-i', 'i-z']
for i, c in enumerate(colors):
    t[c] = mags[:,i]-mags[:,i+1]
    


def masked_nanmean(col):
    return np.nanmean(col[(col<np.inf)&(col>-np.inf)])

def masked_nanstd(col):
    return np.nanstd(col[(col<np.inf)&(col>-np.inf)])
    
grouped_by_wc = t.group_by('Wide Cell Assignment')
averages = grouped_by_wc.groups.aggregate(masked_nanmean)
stddev = grouped_by_wc.groups.aggregate(masked_nanstd)


# ## Calculate the Wide Cell Weights

# In[16]:


norm = np.sum(t['nz_R_weight'])
WC_weight = np.zeros(somres*somres)
for g in grouped_by_wc.groups:
    WC_weight[int(g['Wide Cell Assignment'][0])] = np.sum(g['nz_R_weight'])/norm

with open('../outputs/BFD/cell_weights.pkl', 'wb') as f:
    pickle.dump(WC_weight, f)
    
plt.imshow(WC_weight.reshape((somres,somres)), vmax=0)
plt.colorbar()
plt.title('Wide Cell Weights')
plt.show()


# In[26]:


occupation=np.histogram(t['Wide Cell Assignment'], 
                        bins=somres*somres, 
                        range=(0,somres*somres))[0]
mask = np.where(WC_weight>0)
#occupation[mask] = 0
print(np.median(occupation))

plt.imshow(occupation.reshape((somres, somres)))
plt.title("Occupation for Cells with Negative Weights")
plt.colorbar()
plt.show()


# ## Plots!

# In[11]:


fig, axs = plt.subplots(2,3, figsize=(20,10))

for a,c in zip(axs.T,colors):
    
    im0=a[0].imshow(averages[c].reshape(somres,somres), ) ; a[0].set_title(label=c+' average')
    im1=a[1].imshow(stddev[c].reshape(somres,somres)) ; a[1].set_title(label=c+' standard deviation')
    
    a[0].axis('off')
    a[1].axis('off')
    
    divider = make_axes_locatable(a[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')
    
    divider = make_axes_locatable(a[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')
    
plt.show()


# In[12]:


fig, axs = plt.subplots(2,4, figsize=(20,10))

for i,a,c in zip(range(4), axs.T,list('griz')):
    
    im0=a[0].imshow(averages['SN_'+c].reshape(somres,somres), vmin=-25) 
    im1=a[1].imshow(stddev['SN_'+c].reshape(somres,somres), vmax=40) 
    
    a[0].set_title(label=c+' Signal to Noise')
    a[1].set_title(label=c+' SN standard deviation')
    
    a[0].axis('off')
    a[1].axis('off')
    
    divider = make_axes_locatable(a[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')
    
    divider = make_axes_locatable(a[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')
    
plt.show()


# In[ ]:




