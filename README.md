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

We refer to there and the example jupyter notebook ``Examples.ipynb`` for more information on this project, as it remains under active development.



