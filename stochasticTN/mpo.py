# 
#
#
#
#

import numpy as np
from typing import Any, Optional, List
from stochasticTN.linalg import svd


class MPO:
    """ Defines the MPO class for the stochasticTN package:
    
    Attributes:
    
        * 'MPO.tensors': A list of MPS tensors with length N and index structure:
        
                 2
                 |
            0 -- x -- 3 
                 |
                 1
                 
        * 'MPO.center': Location of orthogonality site of the MPO. Can be moved with MPO.position(site)
        * 'MPO.bond_dimensions: list of local bond dimensions with length N + 1
            
    """
    
    
    def __init__(self, 
                 tensors: List[Any],
                 center: Optional[int] = None,
                 canonicalize: Optional[bool] = False) -> None:
        """Initialize an MPO from a list of tensors
        
        Args:
            tensors: A list of numpy arrays constituting the tensors of the MPS
        """
                    
        self.tensors = tensors
        self.center = center
        if canonicalize:
            self.canonicalize()
        
    
    def __len__(self) -> int:
        return len(self.tensors)
    
    def __iter__(self):
        return iter(self.tensors)
    
    @property
    def bond_dimensions(self) -> List:
        return [self.tensors[0].shape[0]] + [t.shape[3] for t in self]
    
    @property
    def physical_dimensions(self) -> List:
        return [t.shape[1] for t in self]
    
    def position(self, site: int, normalize_SVs: Optional[bool] = False,
                     Dmax: Optional[int] = None, 
                     cutoff: Optional[float] = 0) -> float:
        ''' Puts MPO in canonical form with respect to `site` such that all 
        tensors to the left and to the right of `site` are unitary
        
        Arg:
            site: the site of the center position
            normalize_SVs: Boolean, if True the vector of singular values s is divided by np.sum(s) 
            Dmax: maximal number of singular values to keep
            cutoff: maximal absolute value for the singular values, SVs below this value are dropped
            
        Returns:
            truncation_error: size of the singular values that are dropped.
        '''
        
        if self.center is None:
            raise ValueError('MPO.center is None, cannot shift its position')
            
        truncation_error = 0
        if site == self.center:
            return truncation_error
        
        elif self.center < site:
            for i in range(self.center, site):
                u, s, v, err = svd(self.tensors[i], -1, Dmax, cutoff, normalizeSVs = normalize_SVs)
                self.tensors[i] = u
                sv = np.tensordot(np.diag(s),v, axes= [1,0]) 
                self.tensors[i+1] = np.tensordot(sv, self.tensors[i+1], axes = [1,0])
                truncation_error += err
        else:
            for i in range(self.center,site, -1):
                u, s, v, err = svd(self.tensors[i], 1, Dmax, cutoff,normalizeSVs = normalize_SVs)
                self.tensors[i] = v
                us = np.tensordot(u,np.diag(s), axes= [1,0]) 
                self.tensors[i-1] = np.tensordot(self.tensors[i-1], us, axes = [3,0])
                truncation_error += err
                
        self.center = site
            
        return truncation_error
    
    def canonicalize(self, normalize_SVs: Optional[bool] = True,
                     Dmax: Optional[int] = None, 
                     cutoff: Optional[float] = 0) -> float:
        ''' Puts MPO in canonical form with respect to the left-most site (0) such that all 
        tensors on the right are unitary. Possibly truncates the singular values of the MPO.
        
        Arg:
            normalize_SVs: Boolean, if True the vector of singular values s is divided by np.sum(s) 
            Dmax: maximal number of singular values to keep
            cutoff: maximal absolute value for the singular values, SVs below this value are dropped
            
        Returns:
            truncation_error: size of the singular values that are dropped.
        '''
        
        truncation_error = 0
        n = len(self)
        
        if self.center is None:
            self.center = 0
            
        truncation_error += self.position(n-1, normalize_SVs, Dmax, cutoff)
        truncation_error += self.position(0, normalize_SVs, Dmax, cutoff)
        
        return truncation_error

class SIS_MPO(MPO):
    ''' class for finite MPO for the contact process in one dimension 
    '''
    def __init__(self, N: int, r: float, s: float, 
             driving: Optional[str] = 'boundary',
             omega: Optional[float] = 1) -> None:
        ''' Builds MPO for the tilted Markov generator of the one-dimensional contact process (SIS model)

        Args:
            N: length on MPO
            r: strength of nearest neighbour infection term
            s: tilting parameter, multiplies all off-diagonal terms by exp(s)
            driving: driving protocol, choose from: 
                * 'boundary' for spontaneous occupation of both boundary sites
                * 'right boundary' for spontaneous occupation of the right boundary sites
                * 'left boundary' for spontaneous occupation of the left boundary sites
                * 'absorbing' for spontaneous occupation of a random site if the absorbing empty chain is reached
                * 'spontaneous' for spontaneous occupation of a random empty site with rate 'omega'
                * 'invN' for spontaneous occupation of a random empty site with rate '1/N'
                * 'invN2' for spontaneous occupation of a random empty site with rate '1/N^2'
            omega: the occupation rate for the driving term
        '''
        self.r = r
        self.s = s
               
        d=2
        Ni = np.array([ [0, 0], [0, 1] ])
        Wsi = np.array([ [-1, 0], [np.exp(s), 0] ])
        Wis = np.array([ [0, np.exp(s)], [0, -1] ])
        ID = np.array([[1,0],[0,1]])

        if driving == 'absorbing':
            Vi = np.array([ [1, 0], [0, 0] ])
            Dw = 6
            L = np.zeros([1,d,d,Dw])
            A = np.zeros([Dw,d,d,Dw])
            R = np.zeros([Dw,d,d,1])

            L[0,:,:,0] = Wis
            L[0,:,:,1] = Ni
            L[0,:,:,2] = Wsi
            L[0,:,:,3] = ID
            L[0,:,:,4] = omega*Wsi
            L[0,:,:,5] = Vi

            R[0,:,:,0] = ID
            R[1,:,:,0] = r*Wsi
            R[2,:,:,0] = r*Ni
            R[3,:,:,0] = Wis
            R[4,:,:,0] = Vi
            R[5,:,:,0] = omega*Wsi

            A[:4,:,:,0] = R[:4,:,:,0]
            A[3,:,:,:4] = L[0,:,:,:4]
            A[5,:,:,4:6] = L[0,:,:,4:6]
            A[4:6,:,:,4] = R[4:6,:,:,0]

            tensors = [None]*N
            tensors[0] = L
            for i in range(1,N-1):
                tensors[i] = A
            tensors[N-1] = R

            super().__init__(tensors=tensors)
            return

        elif driving=='boundary':
            omega2 = omega
        elif driving == 'right boundary':
            omega2 = omega
            omega = 0
        elif driving == 'left boundary':
            omega2 = 0
        elif driving == 'invN':
            omega = 1/N
            omega2 = 1/N
        elif driving == 'invN2':
            omega = 1/N**2
            omega2 = 1/N**2
        elif driving == 'spontaneous':
            omega2 = omega
        else:
            omega = 0
            omega2 = 0

        Dw = 4
        L = np.zeros([1,d,d,Dw])
        A = np.zeros([Dw,d,d,Dw])
        R = np.zeros([Dw,d,d,1])

        L[0,:,:,0] = Wis+ omega*Wsi
        L[0,:,:,1] = Ni
        L[0,:,:,2] = Wsi
        L[0,:,:,3] = ID

        R[0,:,:,0] = ID
        R[1,:,:,0] = r*Wsi
        R[2,:,:,0] = r*Ni
        R[3,:,:,0] = Wis + omega2*Wsi

        A[:,:,:,0] = R[:,:,:,0]
        A[3,:,:,:] = L[0,:,:,:]
        if driving == 'boundary' or driving == 'right boundary' or driving == 'left boundary':
            A[3,:,:,0] = Wis
        else:
            pass
        
        tensors = [None]*N
        tensors[0] = L
        tensors[N-1] = R
        for i in range(1,N-1):
            tensors[i] = A

        super().__init__(tensors = tensors)

class network_SIS(MPO):
    '''
    Builds an MPO for the epsilon-SIS Markovian network model 
    '''
    def __init__(self, A, r: float, s: float, 
               epsilon: Optional[float] = 1e-3,
               cutoff: Optional[float] = 1e-12)->None:
        '''
        Docstring
        '''
        self.r = r
        self.s = s
        self.epsilon = epsilon
        
        self.N = np.shape(A)[0]
    
        d=2
        Ni = np.array([ [0, 0], [0, 1] ])
        Wsi = np.array([ [-1, 0], [np.exp(s), 0] ])
        Wis = np.array([ [0, np.exp(s)], [0, -1] ])
        ID = np.array([[1,0],[0,1]])

        L = np.zeros([1,d,d,2])
        R = np.zeros([2,d,d,1])
        M = np.zeros([2,d,d,2])

        L[0,:,:,0] = Wis + epsilon*Wsi
        L[0,:,:,1] = ID
        R[0,:,:,0] = ID
        R[1,:,:,0] = Wis + epsilon*Wsi
        M[:,:,:,0] = R[:,:,:,0]
        M[1,:,:,:] = L[0,:,:,:]


        tensors = self.N*[None]
        tensors[0] = L
        for i in range(1,self.N-1):
            tensors[i] = M
        tensors[self.N-1] = R
        super().__init__(tensors = tensors, center=0, canonicalize=False)
        e = 0

        for i in range(len(self)):
            for j in range(i,len(self)):
                if A[i,j] != 0:
                    self.add_edge(i, j, r, s)
                    e += self.canonicalize(normalize_SVs = False, cutoff = cutoff) 
        
        return
                       
    def add_edge(self,i,j,r,s):
        N = len(self)
        d=2
        Ni = np.array([ [0, 0], [0, 1] ])
        Wsi = np.array([ [-1, 0], [np.exp(s), 0] ])
        Wis = np.array([ [0, np.exp(s)], [0, -1] ])
        ID = np.array([[1,0],[0,1]])
        
        for k in range(len(self)):
            if k < i:
                if k ==0:
                    self.tensors[k] = np.pad(self.tensors[k],((0,0),(0,0),(0,0),(0,1)))
                    self.tensors[k][0,:,:,-1] = ID
                else:
                    self.tensors[k] = np.pad(self.tensors[k],((0,1),(0,0),(0,0),(0,1)))
                    self.tensors[k][-1,:,:,-1] = ID
            elif k == i:
                if k ==0:
                    self.tensors[k] = np.pad(self.tensors[k],((0,0),(0,0),(0,0),(0,2)))
                    self.tensors[k][0,:,:,-2] = Ni
                    self.tensors[k][0,:,:,-1] = r*Wsi
                else:
                    self.tensors[k] = np.pad(self.tensors[k],((0,1),(0,0),(0,0),(0,2)))
                    self.tensors[k][-1,:,:,-2] = Ni
                    self.tensors[k][-1,:,:,-1] = r*Wsi
            elif k > i and k < j:
                self.tensors[k] = np.pad(self.tensors[k],((0,2),(0,0),(0,0),(0,2)))
                self.tensors[k][-2,:,:,-2] = ID
                self.tensors[k][-1,:,:,-1] = ID
            elif k == j:
                if k==N-1:
                    self.tensors[k] = np.pad(self.tensors[k],((0,2),(0,0),(0,0),(0,0)))
                    self.tensors[k][-2,:,:,0] = r*Wsi
                    self.tensors[k][-1,:,:,0] = Ni
                else:
                    self.tensors[k] = np.pad(self.tensors[k],((0,2),(0,0),(0,0),(0,1)))
                    self.tensors[k][-2,:,:,-1] = r*Wsi
                    self.tensors[k][-1,:,:,-1] = Ni
            elif k > j:
                if k == N-1:
                    self.tensors[k] = np.pad(self.tensors[k],((0,1),(0,0),(0,0),(0,0)))
                    self.tensors[k][-1,:,:,0] = ID
                else:
                    self.tensors[k] = np.pad(self.tensors[k],((0,1),(0,0),(0,0),(0,1)))
                    self.tensors[k][-1,:,:,-1] = ID
        return
        
        

class occupancy_MPO(MPO):
    ''' Builds MPO for the occupancy in the MPS whose expectation values gives
        the expected number of occupied sites in the chain 
    '''
    def __init__(self,
                 N: int):
        d=2
        Ni = np.array([ [0, 0], [0, 1] ])
        ID = np.array([[1,0],[0,1]])

        Dw = 2
        L = np.zeros([1,d,d,Dw])
        A = np.zeros([Dw,d,d,Dw])
        R = np.zeros([Dw,d,d,1])

        L[0,:,:,0] = Ni
        L[0,:,:,1] = ID

        R[0,:,:,0] = ID
        R[1,:,:,0] = Ni

        A[:,:,:,0] = R[:,:,:,0]
        A[1,:,:,:] = L[0,:,:,:]

        tensors = [None]*N
        tensors[0] = L
        for i in range(1,N-1):
            tensors[i] = A
        tensors[N-1] = R

        super().__init__(tensors = tensors)
        
class gapMPO(MPO):
    '''Builds MPO measuring the expected value of gaps in occupancy for gapsizes k
    '''
    def __init__(self, N: int, k: int):
        d=2
        Ni = np.array([ [0, 0], [0, 1] ])
        Vi = np.array([ [1, 0], [0, 0] ])
        ID = np.array([[1,0],[0,1]])

        Dw = k+3
        L = np.zeros([1,d,d,Dw])
        A = np.zeros([Dw,d,d,Dw])
        R = np.zeros([Dw,d,d,1])

        L[0,:,:,0] = ID
        L[0,:,:,1] = Ni

        R[-1,:,:,0] = ID
        R[-2,:,:,0] = Ni

        A[:,:,:,-1] = R[:,:,:,0]
        A[0,:,:,:] = L[0,:,:,:]
        for n in range(Dw-3):
            A[n+1,:,:,n+2] = Vi

        tensors = [None]*N
        tensors[0] = L
        for i in range(1,N-1):
            tensors[i] = A
        tensors[N-1] = R
        
        super().__init__(tensors = tensors)