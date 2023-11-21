Tensor Networks for Stochastic Modeling
-------------------------------------

Harvesting the power of tensor networks for representing high-dimensional probability distributions of stochastic systems. 
In this python package, we develop Matrix Product State (MPS) representations for probability distributions and use 
the density matrix renormalization group (DMRG) algorithm to find the leading eigenvectors of Markov generators. 
This allows us to find efficient and accurate representations for the steady state distributions over all
possible configurations for stochastic models, defined by a Markov processes on a chain. The main example where we apply 
these methods is the one-dimensional contact process, or the SIS model to epidemiologists. We use this to study the large
deviation statistics as well as characterize the active-inactive phase transition using information measures in the following paper:
  
  | **Efficient simulations of epidemic models with tensor networks: application to the one-dimensional SIS model**
  | Wout Merbis, Clélia de Mulatier, Philippe Corboz
  | `arxiv/2305.06815 <https://arxiv.org/abs/2305.06815>`_

We refer to there and the example jupyter notebook ``Examples.ipynb`` for more information (for now) as the project remains under active development.
