# Tensor Networks for Probabilistic Modeling

-------------------------------------

Harvesting the power of tensor networks to represent high-dimensional probability distributions as Matrix Product States. 

In this python package, we develop Matrix Product State (MPS) representations for probability distributions over all 
$2^n$ configurations of many-body systems of sites with a binary state space. The main applications are two-fold:

- In ``stochasticTN`` we use the density matrix renormalization group (DMRG) algorithm to find steady state distribution of $2^n$-dimensional Markov generators.
- In ``spinmodelTN`` we construct MPS representations of the $2^n$ dimensional probability distributions of spin models with arbitrary higher-order interactions.

The MPS allows us to find efficient and accurate representations for the steady state distributions over all
$2^n$ configurations of the models. We can use this to efficiently compute observables, characterize the large deviation statistics and perform unbiased sampling of spin configurations.  

One example where we apply these methods is the SIS model of infectious disease spreading. We use this to study rare events, the large
deviation statistics and characterize the active-inactive phase transition using information measures in the following paper:
  
  > **Efficient simulations of epidemic models with tensor networks: application to the one-dimensional SIS model**
  > Wout Merbis, Clélia de Mulatier, Philippe Corboz
  > [arxiv/2305.06815](https://arxiv.org/abs/2305.06815)

We refer to there and the example notebook ``Examples.ipynb`` for more information on the large deviation statistics on the SIS model in one-dimension. In the notebook ``Network_example.ipynb`` we showcase an example based on using DMRG to find the non-equilibrium steady state on networks. 

## Repository contents

### stochasticTN

Code base for performing computations with Matrix Product State (MPS, sometimes called tensor trains) representations of large probability distributions. It is divided into several distinct files

#### `mps.py`
`mps.py` defines the MPS class for the stochasticTN package defining the MPS object with attributes:
- `.tensors` gives a list of the MPS tensors
- `.center` gives the location of the center node in the mixed canonical representation of the mps (all tensors to the left and right of 'center' are unitary matrices w.r.t. contraction from the left/right respectively)
- `.name` optional name for the mps
- `.bond_dimensions` gives a list of bond dimensions of the MPS
- `.physical_dimensions` gives a list of physical dimensions of the MPS
The MPS object is iterable, where iterating the MPS means iterating over MPS.tensors. Its length `len()` is defined as the number of sites in the MPS

`mps.py` contains many methods which are convenient for computations. Most notably:
- `.position(site)` moves the `mps.center` to the specified site by series of singular value decompositions. Specifying `normalize_SVs = True` rescales all singular values to form a distribution which sums to 1. Specifying `Dmax` or `cutoff` truncates the singular value spectrum to maximally `Dmax` SVs with magnitude above `cutoff`
- `.canonicalize()` puts the MPS in canonical form with respect to the left-most site (0) such that all tensors on the right are unitaries. Specifying `normalize_SVs = True` rescales all singular values to form a distribution which sums to 1. Specifying `Dmax` or `cutoff` truncates the singular value spectrum to maximally `Dmax` SVs with magnitude above `cutoff`
- `norm()` returns the norm of the MPS. There are two protocols for computing the norm: `cx = 'stoch'` computes the $$L^1$$ norm (default), `cx = 'complex'` computes the $$L^2$$ norm for complex valued MPS tensors
- `.probabilties()` returns all $$2^n$$ probabilities as computed from the MPS, so this is exponentially costly. Raises a ValueError when `len(MPS)>28`
- `.save('name')` saves the MPS tensors and attributes in local folder "/MPSs/name/"
- `.sample_array()` returns an array sampled according to the probability distribution represented by the MPS

Additionally `mps.py` contains functions for creating some frequently used MPS objects. Most notably:

- `randomMPS(N, D)`: creates a random MPS of length `N` with maximal bond dimension `D`
- `occupied_mps(N,D)`: creates an MPS where all sites are in the occupied state
- `uniform_mps(N,D)`: creates an MPS representing the uniform distribution over all configurations
- `mps_from_array(array)`: creates an MPS with bond dimension 1 from an array of numbers between 0 and 1
- `loadMPS('name')`: loads an MPS from file which was saved using the `.save()` method

#### `mpo.py`




