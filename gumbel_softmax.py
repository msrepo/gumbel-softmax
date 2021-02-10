import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform

np.set_printoptions(suppress=True,precision=2)
shape = (1,1000)

def sample_gumbel(shape,eps =1e-12 ):
    """sample from Gumbel(0,1) Distribution"""
    uni_rv = uniform(size = shape)
    return -np.log(-np.log(uni_rv+eps)+eps)

def standard_gumbel_distribution(x):
    """returns Gumbel(0,1) Distribution where x ~ U(0,1)"""
    return np.exp(-(x + np.exp(-x)))

cat_dist = np.array([0.05,0.05,0.1,0.2,0.2,0.4])
np.testing.assert_almost_equal(np.sum(cat_dist),1.0)

def sample_categorical_distribution_discrete(cat_dist,size):
    """sample from a categorical distribution"""
    np.testing.assert_almost_equal(np.sum(cat_dist),1.0)
    return np.argmax(sample_gumbel((cat_dist.shape[0],size)) + np.log(cat_dist).reshape(-1,1),axis=0)

def sample_categorical_differentiable(cat_dist,size):
    np.testing.assert_almost_equal(np.sum(cat_dist),1.0)
    samples = sample_gumbel((cat_dist.shape[0],size)) + np.log(cat_dist).reshape(-1,1)
    return samples

def apply_softmax_temperature(samples, temperature):
    return np.array([np.exp(samples[:,i]/temperature)/np.exp(samples[:,i]/temperature).sum() for i in range(samples.shape[1])]).T


# plot Gumbel and Categorical Distribution
n_bins = 100
gumbel_rv = sample_gumbel(shape)
fig,axes = plt.subplots(1,2)
axes[0].hist(gumbel_rv.squeeze(),bins=n_bins,density=True,alpha = 0.6, color='red',label='Observed')
x = np.linspace(min(gumbel_rv.squeeze()),max(gumbel_rv.squeeze()),100)
axes[0].plot(x,standard_gumbel_distribution(x),label='Expected')
axes[0].title.set_text('Sampling from a \nGumbel Distribution')
axes[0].legend()

cat_rv = sample_categorical_distribution_discrete(cat_dist,size=1000)
# axes[1].hist(cat_rv.squeeze(),bins=cat_dist.shape[0],density = True)
axes[1].bar(np.arange(cat_dist.shape[0]),cat_dist,alpha=0.6,label='Expected')
y,x = np.histogram(cat_rv,bins = np.arange(-1,cat_dist.shape[0])+0.1,density=True)
axes[1].bar(np.arange(cat_dist.shape[0]),y,color='red',alpha = 0.4, label='Observed')
axes[1].title.set_text('Sampling from a \nCategorical Distribution')
axes[1].legend()
plt.tight_layout()
plt.show()

temps = [0.01,0.1,0.5,1.0,10.]
fig,axes = plt.subplots(1,len(temps),sharex=True,sharey=True)
sample = sample_categorical_differentiable(cat_dist,size=1)
for i,t in zip(np.arange(len(temps)),temps):
    t_sample = apply_softmax_temperature(sample,t)
    print(t_sample.T)
    axes[i].bar(np.arange(cat_dist.shape[0]),t_sample.squeeze(),color='red',alpha=0.4)
    axes[i].title.set_text('t='+str(t))
plt.suptitle('Gumbel-Softmax Distribution \n with various temperature')
plt.tight_layout()
plt.show()