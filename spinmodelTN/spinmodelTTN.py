import numpy as np
import matplotlib.pyplot as plt
from stochasticTN import MPS
from stochasticTN.linalg import svd
from math import ceil
from typing import Any, Optional, List
from copy import deepcopy


class SpinModelTTN():
    '''
    Class for probability distribution of spin models as Tree Tensor Networks. Tensors are all rank three, except the top
    tensor which is of rank 2. The class atribute `hierarchies` determines the number of levels. Tensors are stored in a
    dictionary self.ttn whose keys are the levels (0, ..., self.hierarchies-1) and values are a list of numpy arrays 
    giving the tensors at this level.
    
    Schematically:
    
    physical 
    sites:     0 1 2 3 4 5 6 7 8 9  ... n-1
    
    h = 0:     |_| |_| |_| |_| |_|
               |0| |1| |2| |3| |4|  ... 
                |   |   |   |   |
                
    h = 1:      \___/   \___/   \___/
                |_0_|   |_1_|   |_2_|   ...
                  |       |       |
    
    h = 2:        \_______/
                  |___0___|       ...
                      |
                       
      ...
      
      
    h = self.hierarchies-1:    \_______/
                               |___0___|
    
    Note that for odd n the physical index of the last site does not connect to a tensor at level 0 but at a higher level.
    self.physical_indices returns a list of tuples (h, ih, i) with the location of all physical indices, where h gives 
    the level, ih the index of the tensor at that level and i the physical index on the corresponding tensor.
    I.e. self.ttn[h][ih] is the tensor whose axis i contains the physical index.
    
    
    '''
    def __init__(self, 
                 n: int,
                 mus: List[Any],
                 Js: List[Any],
                 optimize: Optional[bool] = True,
                 Dmax: Optional[int] = None,
                 cutoff: Optional[float] = 0,
                 normalizeSVs: Optional[bool]=False):
        '''
        Constructs a TTN for the probability distributions of a spin model with interactions listed in mus
        and coupling constants listed in Js
        
        Args:
            n: number of spins
            mus: list of arrays with sites present in each interaction (don't give a binary bitstring!)
            Js: coupling strengths of each interaction term
            optimize: if True, use svds to reduce the bond dimensions of the tensors
            Dmax: maximal bond dimensions, if optimize is True, and Dmax is not None, bonds are trunctated if Dmax is reached
            cutoff: cutoff on the singular value spectrum. If optimize is True, SVs below cutoff are dropped
            normalizeSVs: if True, singular values s of each bond are normalized by dividing by sum(s). Prevents the SVs
                from becoming too large or too small.
           
        '''
        if n <=2 :
            raise ValueError( "Need more than 2 physical sites")
            
        if len(mus) != len(Js):
            raise ValueError("mus and Js have different lengths")
            
        self.mus = mus
        self.Js = Js
        self.n = n
        self.hierarchies = ceil(np.log2(self.n))
        self.truncation_error = 0
        self.nttn = None
        self.Httn = None
        
        self.ttn = self.single_term_ttn(0, Dmax = Dmax, cutoff=cutoff, normalizeSVs=normalizeSVs)
        
        
        for i in range(1,len(mus)):
            ttni = self.single_term_ttn(i, Dmax = Dmax, cutoff=cutoff, normalizeSVs=normalizeSVs)
            self.truncation_error += self.product_ttn(ttni,optimize, Dmax, cutoff, normalizeSVs)
            
    def print_tensor_shapes(self):
        '''
        Print the shapes of all tensors in the ttn
        '''
        for h in range(self.hierarchies):
            print('hierarchy' , h, 'with', len(self.ttn[h]), 'tensors')
            for i, t in enumerate(self.ttn[h]):
                print(t.shape)

    
    def norm(self):
        ''' 
        Computes the norm of the TTN by contracting all physical indices with the flat state: [1, 1]
        '''
        length_hierarchies = self.hierarchies
        flat = np.ones(2)
        tensor_list = self.n*[flat]
        for h in range(length_hierarchies):
            new_tensors = []
            for i in range(len(self.ttn[h])):
                k = 2*i
                tens = np.tensordot(self.ttn[h][i], tensor_list[k], axes = [0,0])
                tens = np.tensordot(tens, tensor_list[k+1], axes = [0,0])
                new_tensors.append(tens)
            if len(self.ttn[h])<len(tensor_list)/2:
                new_tensors.append(tensor_list[-1])
            tensor_list = new_tensors
        return float(tens)   
    
   
    def norm_ttn(self):
        '''
        Computes a TTN dictionary whose keys are levels and whose values are the lists of vectors obtained from the ttn tensors 
        contracted against all lower levels and with physical indices summed over. Usefull in the sampling algorithm.      
        '''
        length_hierarchies = self.hierarchies
        flat = np.ones(2)
        tensor_list = self.n*[flat]
        norm_ttn = {}
        for h in range(length_hierarchies):
            new_tensors = []
            for i in range(len(self.ttn[h])):
                k = 2*i
                tens = np.tensordot(self.ttn[h][i], tensor_list[k], axes = [0,0])
                tens = np.tensordot(tens, tensor_list[k+1], axes = [0,0])
                new_tensors.append(tens)
            if len(self.ttn[h])<len(tensor_list)/2:
                new_tensors.append(tensor_list[-1])
            norm_ttn[h] = new_tensors
            tensor_list = new_tensors
        return norm_ttn 
   
    def sample_to_int(self,b):
        ''' Takes string of binary numbers and converts into an integer '''
        return b.dot(1 << np.arange(b.size)[::-1])

    def int_to_sample(self,num, width=None):
        ''' Takes number `num` and returns np.ndarray of the binary representation of this number'''
        return  np.array(list(np.binary_repr(num, width)),dtype=int)
    
    def mps_to_ttn(self, mps_tensors, Dmax = None, cutoff = 0, normalizeSVs=False):
        ''' 
        Maps a list of MPS tensors to a Tree tensor network
        '''
        ttn = {}
        length_hierarchies = ceil(np.log2(self.n))

        for h in range(length_hierarchies-1):
            ttn_current = []
            new_tensors = []
            for i in range(len(mps_tensors)//2):
                n = 2*i
                tens = np.tensordot(mps_tensors[n], mps_tensors[n+1], axes = [-1,0]).transpose(1,2,0,3)
                u, s, v, e = svd(tens,2, Dmax, cutoff, normalizeSVs)
                ttn_current.append(u)
                new_tensors.append(np.tensordot( np.diag(s), v, axes = [1,0]).transpose(1,0,2))
            if n+1<len(mps_tensors)-1:
                new_tensors.append(mps_tensors[-1])
            mps_tensors = new_tensors
            ttn[h] = ttn_current
            
        bond1 = mps_tensors[0].shape[1]
        bond2 = mps_tensors[1].shape[1]
        ttn[length_hierarchies-1] = [np.tensordot(mps_tensors[0], mps_tensors[1], axes = [-1,0]).reshape(bond1,bond2)]
        
        return ttn
    
    def physical_indices(self):
        '''
        Returns a list of tuples containing the physical indices in format (h, ih, i) such that self.ttn[h][ih] is the tensor
        whose axis i contains the physical index index in question.
        '''
        n = self.n
        h = ceil(np.log2(n))
        pi = [(0,i//2,i%2) for i in range(n//2*2)]
        if n%2 == 1:
            hlast = 1
            length = (n-1)//2
            while length>1 and length%2 == 0:
                hlast += 1
                length /= 2

            tens_last = int((length-1)/2)
            pi.append((hlast,tens_last,1))
        return pi
    
    def single_term_ttn(self, term,
                        Dmax: Optional[int]=None,
                        cutoff: Optional[float]=0,
                        normalizeSVs: Optional[bool] = False):
        '''
        Creates the tensors for a single interaction term in the MPS representation and then maps it to a TTN
        
        Args:
            term: indicator for which interaction term
            Dmax: optional maximal bond dimension to keep
            cutoff: optional cutoff on the singular values 
            normalizeSVs: if True divides all singular values s by sum(s)
            
        Returns:
            tensors: TTN dictionary containing the tensors for a single interaction term 
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
            u, s, v, e = svd(t, 1, Dmax, cutoff, normalizeSVs)
            intmpsterms = [u.reshape(1,2,s.shape[0])]
            tnext = np.tensordot(np.diag(s),v, axes = [1,0])
            for i in range(1,len(mu)-1):
                u,s,v,e = svd(tnext, 2, Dmax, cutoff, normalizeSVs)
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

        return self.mps_to_ttn(tensors, Dmax, cutoff, normalizeSVs)
    
    def product_ttn(self, ttn2, optimize=True, Dmax = None, cutoff=0, normalizeSVs=False):
        '''
        Multiplies the tensors in self.ttn by the tensors in ttn2 to create the TTN representation of the 
        product of two TTNs. Overwrites self.ttn by the product.
        
        Args:
            ttn2: the second TTN to multiply against self.ttn
            optimize: if True the singular values are trunctated according to the specified Dmax and cutoff
            Dmax: maximal number of singular values to keep
            cutoff: optional cutoff on the singular values 
            normalizeSVs: if True divides all singular values s by sum(s)
        
        Returns:
            truncation_error: quantity of singular values dropped 
        '''
        n = self.n
        truncation_error = 0
        if n%2 == 1:
            phys_indices = self.physical_indices()
            last_index = phys_indices[-1]

        svs = []
        for h, tensors in self.ttn.items(): 
            if h == 0:
                for i, t in enumerate(tensors):
                    ash = self.ttn[h][i].shape
                    bsh = ttn2[h][i].shape
                    upshape = ash[2]*bsh[2]

                    self.ttn[h][i] = np.pad(self.ttn[h][i], ((0,0),(0,0),(0,upshape-ash[2])))
                    self.ttn[h][i][0,0,:] = np.tensordot( t[0,0,:], ttn2[h][i][0,0,:], axes=((), ())).reshape(upshape)
                    self.ttn[h][i][0,1:] = np.tensordot( t[0,1,:], ttn2[h][i][0,1,:], axes=((), ())).reshape(upshape)
                    self.ttn[h][i][1,0,:] = np.tensordot( t[1,0,:], ttn2[h][i][1,0,:], axes=((), ())).reshape(upshape)
                    self.ttn[h][i][1,1,:] = np.tensordot( t[1,1,:], ttn2[h][i][1,1,:], axes=((), ())).reshape(upshape)

                    if optimize:
                        self.ttn[h][i], s, v, e = svd(self.ttn[h][i],2, Dmax, cutoff, normalizeSVs)
                        truncation_error += e
                        svs.append(np.tensordot(np.diag(s), v, axes =[1,0]))

            elif h < self.hierarchies-1:
                new_svs = []
                for i, t in enumerate(tensors):
                    ash = self.ttn[h][i].shape
                    bsh = ttn2[h][i].shape
                    d1shape = ash[0]*bsh[0]
                    d2shape = ash[1]*bsh[1]
                    upshape = ash[2]*bsh[2]

                    if n%2 == 1 and h == last_index[0] and i == last_index[1]:
                        self.ttn[h][i] = np.pad(self.ttn[h][i], ((0,d1shape-ash[0]),(0,0),(0,upshape-ash[2])))
                        self.ttn[h][i][:,0,:] = np.tensordot( t[:,0,:], ttn2[h][i][:,0,:], 
                                                             axes=((), ())).transpose(0,2,1,3).reshape(d1shape,upshape)
                        self.ttn[h][i][:,1,:] = np.tensordot( t[:,1,:], ttn2[h][i][:,1,:], 
                                                            axes=((), ())).transpose(0,2,1,3).reshape(d1shape,upshape)
                        if optimize:
                            self.ttn[h][i] = np.tensordot(svs[2*i], self.ttn[h][i], axes = [1,0])
                            self.ttn[h][i], s, v, e = svd(self.ttn[h][i],2,Dmax, cutoff, normalizeSVs)
                            truncation_error += e
                            new_svs.append(np.tensordot(np.diag(s), v, axes =[1,0]))

                    else:
                        self.ttn[h][i] = np.pad(self.ttn[h][i], ((0,d1shape-ash[0]),(0,d2shape-ash[1]),(0,upshape-ash[2])))
                        self.ttn[h][i] = np.tensordot(t, ttn2[h][i], 
                                                 axes=((), ())).transpose(0,3,1,4,2,5).reshape(d1shape,d2shape,upshape)
                        if optimize:
                            self.ttn[h][i] = np.tensordot(svs[2*i],self.ttn[h][i],axes = [1,0])
                            self.ttn[h][i] = np.tensordot(svs[2*i+1],self.ttn[h][i],axes = [1,1]).transpose(1,0,2)
                            self.ttn[h][i], s, v, e = svd(self.ttn[h][i],2,Dmax, cutoff, normalizeSVs)
                            new_svs.append(np.tensordot(np.diag(s), v, axes =[1,0]))

                if optimize:
                    if 2*len(new_svs)<len(svs):
                        new_svs.append(svs[-1])
                    svs=new_svs

            else:
                ash = self.ttn[h][0].shape
                bsh = ttn2[h][0].shape
                d1shape = ash[0]*bsh[0]
                d2shape = ash[1]*bsh[1]
                
                if n%2 == 1 and h == last_index[0]:
                    product = np.zeros((d1shape,2))
                    product[:,0] = np.tensordot( self.ttn[h][0][:,0], ttn2[h][0][:,0], 
                                                         axes=((), ())).reshape(d1shape)
                    product[:,1] = np.tensordot( self.ttn[h][0][:,1], ttn2[h][0][:,1], 
                                                        axes=((), ())).reshape(d1shape)
                    self.ttn[h][0]=product
                else:
                    self.ttn[h][0] = np.tensordot(self.ttn[h][0], ttn2[h][0], 
                                         axes=((), ())).transpose(0,2,1,3).reshape(d1shape,d2shape)
                    
                if optimize:
                    self.ttn[h][0] = np.tensordot(svs[0], self.ttn[h][0], axes = [1,0])
                    if n%2 == 1 and last_index[0] == h:
                        pass
                    else:
                        self.ttn[h][0] = np.tensordot(self.ttn[h][0], svs[1], axes = [1,1])
                    
                    truncation_error += self.optimize_topdown(Dmax, cutoff, normalizeSVs)
        return truncation_error
    
    def optimize_topdown(self,Dmax=None, cutoff=0, normalizeSVs=False):
        '''
        Perform a sequence of SVDs on the tensors of the TTN in order to reduce the bond dimensions. 
        Start at the lowest level and SVD up until the highest level.
        
        Args:
            Dmax: maximal number of singular values to keep
            cutoff: optional cutoff on the singular values 
            normalizeSVs: if True divides all singular values s by sum(s)
        
        Returns:
            truncation_error: cumulative quantity of singular values dropped
        '''
        n = self.n
        hierarchies = self.hierarchies
        truncation_error = 0
        svs=[]
        
        phys_indices = self.physical_indices()
        last_index = phys_indices[-1]

        for h in range(hierarchies-1,-1,-1):
#             print("hierachy", h)
            if h == hierarchies-1:
#                 print("Top level matrix svd")
                u,s,v,e = svd(self.ttn[h][0],1, Dmax, cutoff,normalizeSVs)
                truncation_error += e
                self.ttn[h][0] = np.diag(s)
                svs.append(u)
                if last_index[0]==h:
                    self.ttn[h][0] = np.tensordot(self.ttn[h][0], v, axes = [1,0])
                else:
                    svs.append(v.transpose(1,0))
                
            elif h>0:
                new_svs=[]
#                 print("svd at level", h)
                for i, t in enumerate(self.ttn[h]):
                    t = np.tensordot(t, svs[i], axes=[2,0])
                    u,s,v,e=svd(t, 1, Dmax, cutoff,normalizeSVs)
                    truncation_error += e
                    new_svs.append(np.tensordot(u,np.diag(s),axes= [1,0]))
                    if last_index[0]==h and last_index[1] == i:
                        self.ttn[h][i] = v
                    else:
                        u,s,v,e=svd(v.transpose(1,0,2), 1, Dmax, cutoff, normalizeSVs)
                        truncation_error += e
                        new_svs.append(np.tensordot(u, np.diag(s), axes = [1,0]))
                        self.ttn[h][i] = v.transpose(1,0,2)
                    
                    
                if len(self.ttn[h])<len(svs):
                    new_svs.append(svs[-1])
                svs = new_svs
            else:
#                 print("final contractions at level" , h)
                for i, t in enumerate(self.ttn[h]):
                    self.ttn[h][i] = np.tensordot(t, svs[i], axes = (2,0))
        return truncation_error
                
    def optimize_bottomup(self, Dmax=None, cutoff=0, normalizeSVs=False):
        '''
        Perform a sequence of SVDs on the tensors of the TTN in order to reduce the bond dimensions. 
        Start at the highest level and SVD down until the lowest level.
        
        Args:
            Dmax: maximal number of singular values to keep
            cutoff: optional cutoff on the singular values 
            normalizeSVs: if True divides all singular values s by sum(s)
        
        Returns:
            truncation_error: cumulative quantity of singular values dropped
        '''
        n=self.n
        truncation_error = 0
        
        if n%2==1:
            phys_indices = self.physical_indices()
            last_index = phys_indices[-1]
        svs = []
        
        for h, tensors in self.ttn.items():
            if h ==0 :
                for i, t in enumerate(tensors):
                    self.ttn[h][i],s,v,e = svd(t, 2, Dmax, cutoff,normalizeSVs)
                    truncation_error += e
                    svs.append(np.tensordot(np.diag(s),v, axes = [1,0]))

            elif h < self.hierarchies-1:
                new_svs = []
                for i, t in enumerate(tensors):
                    self.ttn[h][i] = np.tensordot(svs[2*i], self.ttn[h][i], axes = [1,0])
                    
                    if n%2 == 1 and h == last_index[0] and i == last_index[1]:
                        pass
                    else:
                        self.ttn[h][i] = np.tensordot(svs[2*i+1],self.ttn[h][i],axes = [1,1]).transpose(1,0,2)
                            
                    self.ttn[h][i], s, v, e = svd(self.ttn[h][i],2,Dmax, cutoff, normalizeSVs)
                    truncation_error += e
                    new_svs.append(np.tensordot(np.diag(s), v, axes =[1,0]))
                
                if 2*len(new_svs)<len(svs):
                    new_svs.append(svs[-1])
                svs=new_svs
            
            else:
                self.ttn[h][0] = np.tensordot(svs[0], self.ttn[h][0], axes = [1,0])
                if n%2 == 1 and last_index[0] == h:
                    pass
                else:
                    self.ttn[h][0] = np.tensordot(self.ttn[h][0], svs[1], axes = [1,1])

        return truncation_error
    
    def optimize(self, Dmax =None, cutoff=0, normalizeSVs=False):
        '''
        Optimize the TTN by performing a sequence of SVDs on the tensors to reduce their bond dimensions. 
        Start at the lowest level and SVD up and then back down until the lowest level.
        
        Args:
            Dmax: maximal number of singular values to keep
            cutoff: optional cutoff on the singular values 
            normalizeSVs: if True divides all singular values s by sum(s)
        
        Returns:
            truncation_error: cumulative quantity of singular values dropped
        '''
        truncation_error = 0
        truncation_error += self.optimize_bottomup(Dmax, cutoff, normalizeSVs)
        truncation_error += self.optimize_topdown(Dmax, cutoff, normalizeSVs)
        return truncation_error
    
    def probabilities(self, norm = 0):
        '''
        Explicitly compute all probabilities from the TTN representation

        Args:
            self: an TTN representation of the energy in the spin basis representation
            norm: optionally give the norm of TTN here, if zero, norm is computed using self.norm() 

        Returns:
            p: a 2**n dimensional vector with the probabilities of all spin configurations 
            
        Raises:
            ValueError if the number of nodes is too large (n>28)
        '''    
        n = self.n
        length_hierarchies = self.hierarchies
        if n > 28:
            raise ValueError("Too many nodes to compute all 2^n probabilities!")
        if norm == 0:
            norm = self.norm()
            
        tens = self.ttn[length_hierarchies-1][0]
        for h in range(length_hierarchies-2,-1,-1):
            for i in range(len(self.ttn[h])):      
                tens = np.tensordot(tens, self.ttn[h][i], axes = [0,2])
            if len(tens.shape)>2*(i+1):
                tp = [k+1 for k in range(len(tens.shape)-1)]
                tp.append(0)
                tens = np.transpose(tens, tp)
        p = np.reshape(tens, 2**n)/norm
        return p
    
    def Hadamard_transform_TTN(self):
        '''
        Performs the Hadamard transform over all physical indices to map the TTN from spin configuration
        basis to binary observable basis.
        
        '''
        H1 = np.array([[1,1],[1,-1]])
        phys_indices = self.physical_indices()
        H_transform = deepcopy(self.ttn)
        
        for h,ih,i  in phys_indices:
            H_transform[h][ih] = np.tensordot(H_transform[h][ih], H1, axes = [i, 0])
            if i ==0 and h==0:
                H_transform[h][ih] = H_transform[h][ih].transpose(2,0,1)
            elif i == 1 and h==0:
                H_transform[h][ih] = H_transform[h][ih].transpose(0,2,1)
            elif h != 0 and h<self.hierarchies-1:
                H_transform[h][ih] = H_transform[h][ih].transpose(0,2,1)
            elif h== self.hierarchies-1:
                pass
            else:
                raise ValueError(f"Option unacounted for physical indices {h}, {ih}, {i}")
        
        return H_transform
            
        
    def observables(self, norm=0):
        '''
        Explicitly compute all observables from the TTN representation

        Args:
            self: an TTN representation of the energy in the spin basis representation
            norm: optionally give the norm of TTN here, if zero, norm is computed using self.norm() 

        Returns:
            obs: a 2**n dimensional vector with all local observables in the binary operator basis   
            
        Raises:
            ValueError if the number of nodes is too large (n>28)
        '''    
        n = self.n
        length_hierarchies = self.hierarchies
        if n > 28:
            raise ValueError("Too many nodes to compute all 2^n probabilities!")
        if norm == 0 :
            norm = self.norm()
        
        if self.Httn == None:
            self.Httn = self.Hadamard_transform_TTN()
        
        tens = self.Httn[length_hierarchies-1][0]
        for h in range(length_hierarchies-2,-1,-1):
            for i in range(len(self.ttn[h])):      
                tens = np.tensordot(tens, self.Httn[h][i], axes = [0,2])
            if len(tens.shape)>2*(i+1):
                tp = [k+1 for k in range(len(tens.shape)-1)]
                tp.append(0)
                tens = np.transpose(tens, tp)
        obs = np.reshape(tens, 2**n)/norm
        return obs
    
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
        
        if self.Httn == None:
            self.Httn = self.Hadamard_transform_TTN()
        if norm == 0 :
            norm = self.norm()
        
        n = self.n
        length_hierarchies = self.hierarchies
        tensor_list = n*[None]
        zero  = np.array([1,0])
        one = np.array([0,1])
        
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
        
#         tensor_list = self.n*[flat]
        for h in range(length_hierarchies):
            new_tensors = []
            for i in range(len(self.Httn[h])):
                k = 2*i
                tens = np.tensordot(self.Httn[h][i], tensor_list[k], axes = [0,0])
                tens = np.tensordot(tens, tensor_list[k+1], axes = [0,0])
                new_tensors.append(tens)
            if len(self.ttn[h])<len(tensor_list)/2:
                new_tensors.append(tensor_list[-1])
            tensor_list = new_tensors
        return float(tens)/norm 
    
    def sample_array(self):
        '''
        Draw a sample spin configuration with statistical weight given by the TTN distribution
        '''

        up=np.array([1,0])
        down=np.array([0,1])
        
        if self.nttn == None:
            self.nttn = self.norm_ttn()
            
        norm_ttn = deepcopy(self.nttn)  

        n = self.n
        samples = []
        samples_tensors = []
        last_updated = self.hierarchies*[0]
        last_site = False

        for i in range(ceil(n/2)):
            if i == n//2:
                norm_ttn[0][i] = np.eye(2)
                last_site=True
            else:
                norm_ttn[0][i] = self.ttn[0][i]
            last_updated[0] = i
    #         print(f'Level {0}, tensor {i} with shape {norm_ttn[0][i].shape}')

            for h in range(1,self.hierarchies):
                ih_index = i//2**h

                if last_updated[h] != ih_index:
                    k = last_updated[h]
    #                 print('update tensor', k, 'at level ', h, 'with shape', norm_ttn[h][k].shape)
                    norm_ttn[h][k] = np.tensordot(norm_ttn[h][k], samples_tensors[-2], axes = [0,-1] )
                    norm_ttn[h][k] = np.tensordot(norm_ttn[h][k], samples_tensors[-1], axes =  [0,-1])

                if ih_index == len(norm_ttn[h])-1 and len(norm_ttn[h])>len(self.ttn[h]):
    #                 print(f"Moving up a tensor into site {ih_index}")
                    norm_ttn[h][ih_index] = norm_ttn[h-1][-1]
                    last_updated[h] = ih_index
                else:
    #                 print(f'Level {h}, tensor {ih_index}, with shape {norm_ttn[h][ih_index].shape}')
                    tens = np.tensordot(self.ttn[h][ih_index], norm_ttn[h-1][2*ih_index], axes = [0,-1] )
                    tens = np.tensordot(tens, norm_ttn[h-1][2*ih_index+1], axes =  [0,-1])


                    if h<self.hierarchies-1:
                        if last_site:
                            norm_ttn[h][ih_index] = tens.transpose(1,0)
                        else:
                            norm_ttn[h][ih_index] = tens.transpose(1,2,0)
                        last_updated[h] = ih_index
                    else:
                        if last_site:
                            pi = tens.reshape(2)
                            pi[pi<0]=0
                            pi/=sum(pi)
    #                         print('Marginal distribution of site',  2*i, pi)

                        else:
                            pi = tens.reshape(4)
                            pi[pi<0] = 0
                            pi /= sum(pi)
    #                         print('Marginal distribution of sites',  2*i, 2*i+1, pi)
            if last_site:
                sample = np.random.choice([0,1],p = pi)
                if sample == 0:
                    samples.append(0)
                    samples_tensors.append(up)
                elif sample == 1:
                    samples.append(1)
                    samples_tensors.append(down)
                else:
                    print("Something not ok")
            else:
                sample = np.random.choice([0,1,2,3],p = pi)
                if sample == 0:
                    samples.extend([0,0])
                    samples_tensors.extend([up,up])
                    norm_ttn[0][i] = np.tensordot(np.tensordot(self.ttn[0][i], up, axes = [0,0]), up, axes = [0,0])
                elif sample == 1:
                    samples.extend([0,1])
                    samples_tensors.extend([up,down])
                    norm_ttn[0][i] = np.tensordot(np.tensordot(self.ttn[0][i], up, axes = [0,0]), down, axes = [0,0])
                elif sample == 2:
                    samples.extend([1,0])
                    samples_tensors.extend([down,up])
                    norm_ttn[0][i] = np.tensordot(np.tensordot(self.ttn[0][i], down, axes = [0,0]), up, axes = [0,0])
                elif sample == 3:
                    samples.extend([1,1])
                    samples_tensors.extend([down,down])
                    norm_ttn[0][i] = np.tensordot(np.tensordot(self.ttn[0][i], down, axes = [0,0]), down, axes = [0,0])
                else:
                    print("Something not ok")

        return np.array(samples)
    
