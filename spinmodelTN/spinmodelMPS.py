import numpy as np
import matplotlib.pyplot as plt
from stochasticTN import MPS
from stochasticTN.linalg import svd
from math import ceil
from typing import Any, Optional, List
from copy import deepcopy

class EnergyMPS(MPS):
    '''
    Class for the MPS representation of the energy for arbitrary spin models
    
    '''
    def __init__(self, 
                 n: int,
                 mus: List[Any],
                 Js: List[Any],
                 optimize: Optional[bool] = True,
                 cutoff: Optional[float] = 0) -> None:
        
        self.n = n
        self.mus = mus
        self.Js = Js

        gmps = self.individual_g(mus[0],Js[0],n)
        for k in range(1,len(mus)):
            gnew = self.individual_g(mus[k],Js[k],n)
            gmps = self.add_operator(gmps,gnew)

        emps = self.compute_eMPS(gmps)
        super().__init__(tensors = emps, canonicalize=False)

        if optimize:
            self.canonicalize(normalize_SVs = False, cutoff=cutoff)
            
          

    def individual_g(self,mu, J, n):
        '''
        Computes MPS representation of a single interaction term in the binary basis. By convention, the 
        coupling strength is multiplies the first site in the interaction term.

        Args:
            mu: list of sites present in the interaction (don't give a binary bitstring!)
            J: coupling strength of interaction term
            n: number of spins

        Returns:
            g: list of MPS tensors of shape (1,2,1)

        '''
        one = np.array([0.,1.]).reshape(1,2,1)
        zero = np.array([1.,0.]).reshape(1,2,1)
        g = [None]*n
        for i in range(n):
            if i in mu:
                g[i] = one
            else:
                g[i] = zero
        g[min(mu)] = J*g[min(mu)]
        return g


    def compute_eMPS(self, gMPS):
        '''
        Maps a g-vector in MPS representation into the energy of all spin configuration and returns 
        the corresponding energy MPS

        Args:
            gMPS: an MPS representation of g in a binary basis

        Returns:
            emps: an MPS representation of the energy in the spin basis representation

        ''' 
        n = len(gMPS)
        H1 = np.array([[1,1],[1,-1]])
        emps = [None]*n
        for i in range(n):
            emps[i] = np.tensordot(gMPS[i], H1, axes = (1,0)).transpose(0,2,1)
        return emps

    def energies(self):
        '''
        Explicitly compute all energies from the MPS representation

        Args:
            emps: an MPS representation of the energy in the spin basis representation

        Returns:
            e: a 2**n dimensional vector the energies of all spin configurations 
        '''    
        
        n = len(self)
        if n > 28:
            raise ValueError("Too many nodes to compute all 2^n energies!")
        e = np.ones(1)
        for i in range(n):
            e = np.tensordot(e, self.tensors[i], axes = (-1,0))
        return e.reshape(2**n)

    def add_operator(self, g, g2):
        n = len(g)
        n2 = len(g2)
        if n != n2:
            raise ValueError("operators of different lengths cannot be added (for now)")

        gsum = [None]*n
        for i in range(n):
            g2sh = g2[i].shape

            if i == 0:
                gsum[i] = np.pad(g[i], ((0,0),(0,0),(0,g2sh[2])))
                gsum[i][:,:,-g2sh[2]:] = g2[i]
            elif i == n-1:
                gsum[i] = np.pad(g[i], ((0,g2sh[0]),(0,0),(0,0)))
                gsum[i][-g2sh[0]:,:,:] = g2[i]
            else:
                gsum[i] = np.pad(g[i], ((0,g2sh[0]),(0,0),(0,g2sh[2])))
                gsum[i][-g2sh[0]:,:,-g2sh[2]:] = g2[i]

        return gsum

class SpinModelMPS(MPS):
    '''
    Class for probability distribution of spin models as MPS
    '''
    def __init__(self, 
                 n: int,
                 mus: List[Any],
                 Js: List[Any],
                 Dmax: Optional[int] = None,
                 optimize: Optional[bool] = True,
                 cutoff: Optional[float] = 0) -> None:
        '''
        Constructs an stochasticMPS for the probability distributions of a spin model with interactions listed in mus
        and couling constants listed in Js
        
        Args:
            n: number of spins
            mus: list of arrays with sites present in each interaction (don't give a binary bitstring!)
            Js: coupling strengths of each interaction term
            
        '''
        self.mus = mus
        self.Js = Js
        self.n = n
        self.Rs = None
        
        if len(mus) != len(Js):
            raise ValueError("mus and Js have different lengths")
       
        tens0 = self.single_term_mps(0, Dmax = Dmax, cutoff=cutoff)
        super().__init__(tensors = tens0, canonicalize=False)
        
        for i in range(1,len(mus)):
            tensi = self.single_term_mps(i, Dmax = Dmax, cutoff=cutoff)
            self.product_mps(tensi, optimize=optimize, Dmax = Dmax, cutoff=cutoff)
            
        
    def compute_Rs(self):
        n = len(self)
        flat = np.ones(2)
        self.Rs = {n-1: np.ones(1)}
#         self.Rs = {n-2: np.tensordot(self.tensors[n-1], flat, axes = [1,0])}
        for i in range(n-2,-1,-1):
            keti = np.tensordot(self.tensors[i+1], flat, axes = [1,0])
            self.Rs[i] = np.tensordot(keti, self.Rs[i+1], axes = [1,0])

    
    def sample_to_int(self,b):
        ''' Takes string of binary numbers and converts into an integer '''
        return b.dot(1 << np.arange(b.size)[::-1])

    def int_to_sample(self,num, width=None):
        ''' Takes number `num` and returns np.ndarray of the binary representation of this number'''
        return  np.array(list(np.binary_repr(num, width=width)),dtype=int)

    def single_term_mps(self,term,
                        Dmax: Optional[int]=None,
                        cutoff: Optional[float]=0):
        '''
        Create the tensors for a single interaction term in the MPS representation
        
        Args:
            term: indicator for which interaction term
            Dmax: optional maximal bond dimension to keep
            cutoff: optional cutoff on the singular values 
            
        Returns:
            tensors: list of np.ndarrays containing the MPS tensors for the single site
        '''
        
        n = self.n
        mu = self.mus[term]
        J = self.Js[term]
        if len(mu)>n:
            raise ValueError("More spins in the interaction term than n")
        elif len(mu)>25:
            raise ValueError("Too many spins in the interaction")
        
        if type(mu) != set:
            mu = np.sort(mu)
        
        tensors = n*[None]
        
        if len(mu)>0:
            t = np.zeros(2**len(mu))
            for i in range(2**len(mu)):
                sign = 1-2*(np.sum(self.int_to_sample(i))%2)
                t[i] = np.exp(sign*J)
            t = t.reshape(len(mu)*(2,))

        if len(mu) == 1:
            intmpsterms = [t.reshape(1,2,1)]
        elif len(mu) == 0:
            pass
        else:
            u, s, v, e = svd(t, 1, normalizeSVs=True, Dmax = Dmax, cutoff=cutoff)
            intmpsterms = [u.reshape(1,2,s.shape[0])]
            tnext = np.tensordot(np.diag(s),v, axes = [1,0])
            for i in range(1,len(mu)-1):
                u,s,v,e = svd(tnext, 2, normalizeSVs=True, Dmax = Dmax, cutoff=cutoff)
                intmpsterms.append(u)
                tnext = np.tensordot(np.diag(s),v, axes = [1,0])
            intmpsterms.append(tnext.reshape(s.shape[0],2,1))
        
        mucounter = 0
        curD = 1
        for i in range(n):
            if i in mu:
                tensors[i] = intmpsterms[mucounter]
                curD = intmpsterms[mucounter].shape[2]
                mucounter += 1

            else:
                tensors[i] = np.zeros((curD,2,curD))
                tensors[i][:,0,:] = np.eye(curD)
                tensors[i][:,1,:] = np.eye(curD)

        return tensors
    
    def product_mps(self, tensors, optimize=True, Dmax = None, cutoff=0):
        n = self.n
        n2 = len(tensors)
        if n != n2:
            raise ValueError("MPS's of different lengths")


        for i, t in enumerate(self.tensors):
            ash = t.shape
            bsh = tensors[i].shape
            lshape = ash[0]*bsh[0]
            rshape = ash[2]*bsh[2]
            
            
            self.tensors[i] = np.pad(self.tensors[i], ((0,lshape-ash[0]),(0,0),(0,rshape-ash[2])))
            self.tensors[i][:,0,:] = np.tensordot( t[:,0,:], tensors[i][:,0,:], 
                                             axes=((), ())).transpose(0,2,1,3).reshape(lshape,rshape)
            self.tensors[i][:,1,:] = np.tensordot( t[:,1,:], tensors[i][:,1,:], 
                                             axes=((), ())).transpose(0,2,1,3).reshape(lshape,rshape)
            if optimize:
                if i>0:
                    self.tensors[i] = np.tensordot(sv, self.tensors[i], axes = [1,0])
                
                if i<n-1:
                    self.tensors[i], s, v , _ = svd(self.tensors[i], 2, Dmax, cutoff)
                    sv = np.tensordot(np.diag(s), v, axes = [1,0])
                    self.center = i+1
            
            
        if optimize:
            self.position(0, cutoff = cutoff, Dmax = Dmax)

        return
    
    def probabilities(self, norm = 0):
        '''
        Explicitly compute all probabilities from the MPS representation

        Args:
            mps: an MPS representation of the probability in the spin basis representation

        Returns:
            p: a 2**n dimensional vector with the probabilities of all spin configurations 
        '''    
        if norm == 0:
            norm = self.norm()
            
        n = len(self)
        if n > 28:
            raise ValueError("Too many nodes to compute all 2^n probabilities!")
        p = self.tensors[0]
        for i in range(1,n):
            p = np.tensordot(p, self.tensors[i], axes = (-1,0))
        return p.reshape(2**n)/norm
    
    def observables(self, norm=0):
        '''
        Explicitly compute all observables from the MPS representation

        Args:
            emps: an MPS representation of the probability in the spin basis representation

        Returns:
            p: a 2**n dimensional vector with the observables of all spin configurations 
        '''    
        
        if norm==0:
            norm = self.norm()
        
        H1 = np.array([[1,1],[1,-1]])
#         H1 = np.linalg.inv(H1)
        n = len(self)
        if n > 28:
            raise ValueError("Too many nodes to compute all 2^n probabilities!")
        p = np.tensordot(self.tensors[0], H1, axes = [1,0]).transpose(0,2,1)
        for i in range(1,n):
            Htransformi = np.tensordot(self.tensors[i], H1, axes = [1,0]).transpose(0,2,1)
            p = np.tensordot(p, Htransformi, axes = (-1,0))
            
        return p.reshape(2**n)/norm
    
    def compute_observable(self, obs, norm = 0):
        ''' 
        Computes the expectation value of a single observable
        
        Args:
            obs: the observable. 
                if integer: the binary representation will be the observable
                if list or array: the observable is the concatenation of spin sites specified in the list
                if binary string: the observable contains all sites corresponding to the 1s in the string
            norm: optionally give the norm of TTN here, if zero, norm is computed using self.norm() 
        '''

        if norm == 0 :
            norm = self.norm()
        
        n = self.n
        tensor_list = n*[None]
        zero  = np.array([1,1])
        one = np.array([1,-1])
        
        if type(obs) == int:
            obs_ls = self.int_to_sample(obs, n)
        elif type(obs) == str:
            obs_ls = np.array([int(bit) for bit in obs])
        else:
            obs_ls = np.zeros(n)
            for site in obs:
                obs_ls[site] = 1
        
        for i in range(n):
            if obs_ls[i] == 0:
                tensor_list[i] = zero
            elif obs_ls[i] == 1:
                tensor_list[i] = one
            else:
                print("OOPS!")
        
        tens = np.tensordot(self.tensors[0], tensor_list[0], axes = [1,0])
        for i in range(1,n):
            tens = np.tensordot(tens, self.tensors[i], axes = [-1,0])
            tens = np.tensordot(tens, tensor_list[i], axes = [1,0])
        
        return float(tens)/norm 
    
#     def sample_array(self):
#         n = len(self)
#         flat = np.ones(2)
#         up=np.array([1,0])
#         down=np.array([0,1])
        
#         if self.Rs == None:
#             self.compute_Rs()
#         else:
#             pass

#         sample = []
#         Ls = {}

#         p0 = np.tensordot(self.tensors[0], self.Rs[0], axes = [2,0])
#         p0[p0<0]=0
#         p0 = p0.reshape(2)/np.sum(p0)
        
#         sample.append(np.random.choice([0,1],p = p0))
#         if sample[0] == 0:
#             Ls[1] = np.tensordot(self.tensors[0],up, axes = [1,0])
#         else:
#             Ls[1] = np.tensordot(self.tensors[0],down, axes = [1,0])

#         for i in range(1,n-1):
#             pi = np.tensordot(np.tensordot(Ls[i],self.tensors[i],axes = [1,0]),self.Rs[i], axes = [2,0])
#             pi = pi.reshape(2)
#             pi[pi<0] = 0
#             pi /= sum(pi)
#             sample.append(np.random.choice([0,1],p = pi))
#             if sample[i] == 0:
#                 Ls[i+1] = np.tensordot(Ls[i], np.tensordot(self.tensors[i],up, axes = [1,0]), axes = [1,0])
#             else:
#                 Ls[i+1] = np.tensordot(Ls[i], np.tensordot(self.tensors[i],down, axes = [1,0]), axes = [1,0])

#         pn = np.tensordot(Ls[n-1], self.tensors[n-1], axes = [1,0])
#         pn = pn.reshape(2)
#         pn[pn<0]=0
#         pn /= sum(pn)
#         sample.append(np.random.choice([0,1],p = pn))

#         return np.array(sample)
    
    def sample_array(self,batch_size=6):
        n = len(self)

        num_batches = n//batch_size
        remainder = n%batch_size

        flat = np.ones(2)
        up=np.array([1,0])
        down=np.array([0,1])

        if self.Rs == None:
            self.compute_Rs()
        else:
            pass

        sample = []
        Ls = {0: np.ones(1)}

        for x in range(num_batches):
            i = x*batch_size
            p = np.tensordot(Ls[x*batch_size], self.tensors[x*batch_size], axes = [-1,0])
            for y in range(1,batch_size):
                i += 1
                p = np.tensordot(p, self.tensors[i], axes = [-1,0])
            if i < n-1:
                p = np.tensordot(p, self.Rs[i], axes = [-1,0])
            p[p<0]=0
            p = p.flatten()/np.sum(p)

            choice = np.random.choice([i for i in range(2**batch_size)],p = p)
            sample.extend(self.int_to_sample(choice,width=batch_size))

            for y in range(batch_size):
                i = x*batch_size + y
                temp = np.tensordot(Ls[i], self.tensors[i], axes = [-1,0])

                if sample[i] == 0:
                    Ls[i+1] = np.tensordot(temp, up, axes = [0,0])
                else:
                    Ls[i+1] = np.tensordot(temp,down, axes = [0,0])

        if remainder != 0:
            i = num_batches*batch_size
            p = np.tensordot(Ls[i], self.tensors[i], axes = [-1,0])
            for y in range(1,remainder):
                i +=  1
                p = np.tensordot(p, self.tensors[i], axes = [-1,0])
            if i < n-1:    
                p = np.tensordot(p, self.Rs[i], axes = [-1,0])
            p[p<0]=0
            p = p.flatten()/np.sum(p)

            choice = np.random.choice([i for i in range(2**remainder)],p = p)
            sample.extend(self.int_to_sample(choice,width=remainder))


        return np.array(sample)
    
    def sample_array2(self,num):
        '''
        Sample array of bits from MPS
        
        Args:
            num: number of samples required
        
        Returns:
            sample: numpy array of bits drawn according to the MPS representation of the probability distribution
        '''
        n = len(self)
        flat = np.ones(2)
        up=np.array([1,0])
        down=np.array([0,1])
        bonds = self.bond_dimensions
        
        if self.Rs == None:
            self.compute_Rs()
        else:
            pass

        samples = np.zeros((num, n))
        Ls = {}
        Ls[1] = np.zeros((num,bonds[0],bonds[1]))
        
        p0 = np.tensordot(self.tensors[0], self.Rs[0], axes = [2,0])
        p0[p0<0]=0
        p0 = p0.reshape(2)/np.sum(p0)
        
        oneup = np.tensordot(self.tensors[0],up, axes = [1,0])
        onedown = np.tensordot(self.tensors[0],down, axes = [1,0])
        
        for no in range(num):
            bit = np.random.choice([0,1],p = p0)
            samples[no,0] = bit
            if bit == 0:
                Ls[1][no] = oneup
            else:
                Ls[1][no] = onedown
                
        for i in range(1,n-1):
            iup = np.tensordot(self.tensors[i],up, axes = [1,0])
            idown = np.tensordot(self.tensors[i],down, axes = [1,0])
            Ls[i+1] = np.zeros((num,bonds[0],bonds[i+1]))
            
            for no in range(num):
                pi = np.tensordot(np.tensordot(Ls[i][no],self.tensors[i],axes = [1,0]),self.Rs[i], axes = [2,0])
                pi = pi.reshape(2)
                pi[pi<0] = 0
                pi /= sum(pi)
                bit = np.random.choice([0,1],p = pi)
                samples[no,i] = bit
                if bit == 0:
                    Ls[i+1][no] = np.tensordot(Ls[i][no], iup, axes = [1,0])
                else:
                    Ls[i+1][no] = np.tensordot(Ls[i][no], idown, axes = [1,0])

        for no in range(num):
            pn = np.tensordot(Ls[n-1][no], self.tensors[n-1], axes = [1,0])
            pn = pn.reshape(2)
            pn[pn<0]=0
            pn /= sum(pn)
            bit = np.random.choice([0,1],p = pn)
            samples[no,n-1] = bit 

        return samples
                
    
def randomSM(n, K, optimize = True, cutoff = 0):
    mus = K*[None]
    Js = K*[None]
    for k in range(K):
        mus[k] = np.random.choice([i for i in range(n)],np.random.randint(1,n), replace=False)
        Js[k] = np.random.uniform(-1,1)
    return EnergyMPS(n,mus, Js, optimize=optimize, cutoff=cutoff)


