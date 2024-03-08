import numpy as np
import matplotlib.pyplot as plt
from stochasticTN import MPS
from stochasticTN.linalg import svd
import networkx as nx
from scipy.sparse.linalg import eigsh
from typing import Any, Optional, List

# Collecting useful functions and operations for the Spin Model Tensor network library here

def generate_random_pairwise_model(n, k, beta=0):
    '''
    Generate a random pairwise spin model
    
    Args:
        n: Number of spins
        k: Number of interaction terms
        beta: temperature, if left unspecified all Js will be chosen at random
    
    Returns
        mus: List of interaction terms indicating which spins appear in the term 
        Js:  List of coupling constants 
    
    '''
    mus = k*[None]
    
    if beta !=0:
        Js = k*[beta]
    else:
        Js = k*[None]
       
    for mu in range(k):
        mus[mu] = np.random.choice([i for i in range(n)], 2, replace = False)
        if beta == 0:
            Js[mu] = np.random.uniform(-1,1)
        
    return mus, Js

def generate_random_spinmodel(n, k, max_order = 8, beta = 0):
    '''
    Generate a random spin model with higher order terms up to max_order
    
    Args:
        n: Number of spins
        k: Number of interaction terms
        max_order: maximal order of the interaction terms
        beta: temperature, if left unspecified all Js will be chosen at random
    
    Returns
        mus: List of interaction terms indicating which spins appear in the term 
        Js:  List of coupling constants 
    
    '''
    mus = k*[None]
    if beta == 0:
        Js = k*[None]
    else:
        Js = k*[beta]
        
    for mu in range(k):
        mus[mu] = set([np.random.randint(0,n) for i in range(np.random.randint(2,max_order+1))])
        if beta == 0:
            Js[mu] = np.random.uniform(-1,1)
    return mus, Js

def generate_spin_model_from_graph(G, beta):
    '''
    Generate the interaction terms for a pairwise spin model from a networkx graph G
    
    Args:
        G: networkx graph object
        beta: temperature
    
    Returns
        mus: List of interaction terms indicating which spins appear in the term 
        Js:  List of coupling constants 
    
    '''
    mus = list(G.edges)
    Js = [beta for _ in range(len(mus))]
    return mus, Js

def empirical_dist(samples, n):
    ''' Create empirical distribution from a list of spin samples
    
    Args:
        samples: list of spin samples as integers
        n: number of spins
        
    Returns:
        emp_dist: empirical distribution over all 2**n spin configurations
    '''
    emp_dist = np.zeros(2**n)
    ns, c = np.unique(samples, return_counts = True)
    emp_dist[ns] = c / np.sum(c)
    return emp_dist

def sample_to_int(b):
    ''' Takes string of binary numbers and converts into an integer '''
    return b.dot(1 << np.arange(b.size)[::-1])

def int_to_sample(num, width=None):
    ''' Takes number `num` and returns np.ndarray of the binary representation of this number'''
    return  np.array(list(np.binary_repr(num, width=width)),dtype=int)


def mus_to_matrix(mus, n):
    ''' Converts a list of interaction terms mus to a matrix'''
    matrix = np.zeros((n,len(mus)), dtype=int)
    for m, mu in enumerate(mus):
        if type(mu) == set:
            mu = list(mu)
        matrix[mu, m] = 1
    return matrix

def matrix_to_mus(matrix):
    ''' Inverse of mus_to_matrix, converts a matrix back into a list of interaction terms'''
    if type(matrix) != np.ndarray:
        matrix = np.array(matrix)
    
    n, m = matrix.shape
    mus = []
    for i in range(m):
        mus.append([k for bit, k in zip(matrix[:,i],range(n)) if bit==1])
    return mus

def row_reduction(input_mat):
    '''Function to perform row reduction over F2 (binary field) for a NumPy array
    
    Returns the row-reduced matrix and the basis transformation
    '''

    matrix = input_mat.copy()
    
    if type(matrix) == list:
        matrix= np.array(matrix)
    num_rows, num_cols = matrix.shape

    lead = 0
    basis_transformation = np.eye(num_rows, dtype=int)
    for r in range(num_rows):
        if lead >= num_cols:
            return matrix, basis_transformation
        
        i = r
        while matrix[i, lead] == 0:
            i += 1
            if i == num_rows:
                i = r
                lead += 1
                if num_cols == lead:
                    return matrix, basis_transformation
        
        # Swap the rows (if needed) and update basis transformation
        matrix[[i, r]] = matrix[[r, i]]
        basis_transformation[[i, r]] = basis_transformation[[r, i]]
        
        # Perform row operations
        for i in range(num_rows):
            if i != r and matrix[i, lead] == 1:
                matrix[i] ^= matrix[r]  # XOR operation (addition in F2)
                basis_transformation[i] ^= basis_transformation[r]  # Update basis transformation
        
        lead += 1
    
    return matrix, basis_transformation

def get_maps(matrix, result, basis_transformation):
    ''' Create a dictionary of the forward and backwards spin basis transformation
    '''
    num_rows, num_cols = matrix.shape

    backward_map = {i: np.where(trans==1)[0] for i, trans in enumerate(basis_transformation.transpose())}

    
    forward_map = {}
    num = 0
    col = 0
    while num<num_rows and col<num_cols:
        non_zero_count = np.count_nonzero(result[:, col])
        if non_zero_count == 1 and result[num,col]==1:
            forward_map[num] = np.where(matrix[:,col] == 1)[0]
            num += 1
        col+=1
    
    if len(forward_map)<len(backward_map):
        for key, val in backward_map.items():
            if len(val)==1 and val[0] not in forward_map:
                forward_map[val[0]] = np.array([key])
    
    return forward_map, backward_map


def mus_to_L(n, mus, Js):
    ''' Create a graph Laplacian out of a set of interaction terms mus and their couplings Js'''
    L = np.zeros((n,n))
    for n, mu in enumerate(mus):
        for i in mu:
            for j in mu:
                if i!=j:
                    L[i,j] += -abs(Js[n])/len(mu)
    d = -np.sum(L, axis=0)    
    L += np.diag(d)
    return L


def find_permutation_connected(L):
    ''' Find the permutation which minimizes the distance in a one-dimensional projection of the spin model  
    '''
    n = L.shape[0]
    evs, evex = eigsh(L, 2, which='SM')
    FiedlerEV = evex[:,1]
    permutation  = [i for _,i in sorted(zip(FiedlerEV.real,range(n)))]
    
    return permutation

def find_permutation_nx(G):
    ''' Find the optimal permutation based on a networkx graph G
    '''
    n = len(G)
    permutation = []
    for component in nx.connected_components(G):
        component = list(component)
#         print(component)
        
        if len(component) == 1:
            permutation.extend(component)
        else:
            Lcomponent = nx.laplacian_matrix(G, nodelist=component)
#             print(Lcomponent.shape)
            evs, evex = eigsh(Lcomponent, 2, which='SM')
            FiedlerEV = evex[:,1]
#             print(sorted(FiedlerEV))
            permutation.extend([i for _,i in sorted(zip(FiedlerEV.real,component))])
    
    return permutation


def Fiedlermapping(n,mus, Js):
    '''Find the permutation which maps to the optimal ordering of spins in the MPS 
    '''
    L = mus_to_L(n, mus, Js)
    
    A = -L.copy()
    for i in range(n):
        A[i,i] = 0
    G = nx.from_numpy_matrix(A)
    connected = nx.is_connected(G)
    if connected:
        permutation = find_permutation_connected(L)
    else:
        permutation = find_permutation_nx(G)
    
#     for_map = {i: val for i, val in enumerate(permutation)}
    perm_map = {val: i for i, val in enumerate(permutation)}
    
    return permutation, perm_map

def optimize_spin_basis(n, mus, Js):
    ''' Optimize the spin basis by first performing a gauge transformation to the minimally complex model
    and then optimizing the order of the transformed spins to minimize the correlation between them.
    '''
    #Row reduce interactions to new spin basis
    matrix = mus_to_matrix(mus, n)
    result, basis_trafo = row_reduction(matrix)
    fw_map, bw_map = get_maps(matrix, result, basis_trafo)
    muprimes = matrix_to_mus(result)
    
    #optimize ordering of nodes in new basis
    perm, Fmap = Fiedlermapping(n,muprimes,Js)

    mu_finals = []
    for mu in muprimes:
        mu_finals.append(np.vectorize(Fmap.get)(mu))
    
    forward_map = {}
    for s, sigma in fw_map.items():
        spin = Fmap[s]
        forward_map[spin] = sigma
        
    backward_map = {}
    for s, sigma in bw_map.items():
        backward_map[s] = np.vectorize(Fmap.get)(sigma)
    
    return mu_finals, forward_map, backward_map


def perm_to_mus(mus,perm):
    '''
    Permute the interaction terms to the new ordering of spins
    '''
    new_mus = []
    perm_map = {site: i for i, site in enumerate(perm)}
    for mu in mus:
        new_mus.append(np.vectorize(perm_map.get)(list(mu)))
    return new_mus


def permute2n(bw_map):
    '''
    Constructs 2^n dimensional permutation matrix based on permutation of spins specified in the mapping `bw_map`
    
    Don't use for large n :)
    
    '''
    n = len(bw_map)
    perm2n = np.zeros(2**n, dtype = int)
    for num in range(2**n):
        b = np.array(list(np.binary_repr(num, width=n)),dtype=int)
        s = 1-2*b
        c = np.array([np.product(s[bw_map[i]]) for i in range(n)])
        c = (1/2*(1-c)).astype(int)
        number = c.dot(1 << np.arange(c.size)[::-1])
        perm2n[number] = num
    return perm2n    