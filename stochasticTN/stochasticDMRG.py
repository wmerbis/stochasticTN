# Copyright 2023 stochasticTN Developers, GNU GPLv3

''' Implementation of variational MPS optimization for (titlted) Markov generators in MPO form '''

import numpy as np
from scipy.sparse.linalg import eigs, LinearOperator, ArpackNoConvergence
from sys import stdout
from stochasticTN.mps import MPS
from stochasticTN.mpo import MPO, occupancy_MPO
from stochasticTN.operations import MPOexpectation, MPOvariance
from stochasticTN.linalg import svd
from typing import Any, Text, Union, Optional
import time
Tensor = np.ndarray

class DMRG:
    """ Class for running the DMRG algorithm for stochastic MPS, training for the 
        largest real eigenvector of a Markov chain expressed as an MPO
    """
    
    def __init__(self, mps: MPS, mpo: MPO,
                 left_bdy: Tensor,
                 right_bdy: Tensor):
        """ Class for DMRG simulations

        Args:
            mps: initial MPS object 
            mpo: an MPO object whose leading eigenvector is sought
            left_bdy: the left boundary environment
            right_bdy: the right boundary environment
        """
        if len(mps) != len(mpo):
            raise ValueError(f'MPS and MPO are of different lengths: {len(mps)} and {len(mpo)} ')

        if mps.physical_dimensions != mpo.physical_dimensions:
            raise ValueError(f'MPS and MPO have different physical dimensions')

        self.mps = mps
        self.mpo = mpo
        self.Ls = {0: left_bdy}
        self.Rs = {len(mps) - 1: right_bdy}
    
   
    def add_R(self, site):
        ''' Adds a single right environment for `site` given that the environment to the 
            right of `site` is known.
        '''
        nextR = np.tensordot(self.mps.tensors[site+1],self.Rs[site+1], axes=[2,0])
        nextR = np.tensordot(self.mpo.tensors[site+1],nextR, axes = ([2,3],[1,2]))
        nextR = np.tensordot(np.conj(self.mps.tensors[site+1]),nextR, axes = ([1,2],[1,3]))
        self.Rs[site] = np.transpose(nextR, (2,1,0))
        
    def compute_Rs(self):
        ''' Compute the right environments for each site right off (and including) 
            mps.center
        '''
        N=len(self.mps)
        site = self.mps.center
        
        for i in range(N-2,site-1,-1):
            self.add_R(i)
    
    def add_L(self,site):
        ''' Adds a single left environment for `site` given that the environment to the 
            left of `site` is known.
        '''
        nextL = np.tensordot(self.Ls[site-1], self.mps.tensors[site-1], axes = [0,0])
        nextL = np.tensordot(nextL, self.mpo.tensors[site-1], axes = ([0,2],[0,2]))
        self.Ls[site] = np.tensordot(nextL, np.conj(self.mps.tensors[site-1]), axes = ([0,2],[0,1]))
        
    def compute_Ls(self):
        ''' Compute the left environments for each site left off (and including) 
            mps.center
        '''
        N=len(self.mps)
        site = self.mps.center
        
        for i in range(1,site+1):
            self.add_L(i)  
            
                             
    def single_site_matvec(self, vec: np.ndarray,
                        R: np.ndarray,
                        L: np.ndarray,
                        MPOmat: np.ndarray) -> np.ndarray:
        ''' Implements the single site matrix-vector multiplication needed for the 
            single site DMRG routine
            
        Args:
            vec: single site 'ket' flattened to a vector to be multiplied by MPO
            R: right environment of 'ket'
            L: left environment of 'ket'
            MPOmat: local MPO matrix to multiply against 'ket'
            
        Returns:
            Wv: vector which is the result of the multiplication of vec with the effective single
                site generator
        '''
        
        Rsh=R.shape
        Lsh=L.shape
        ket = np.reshape(vec, (Lsh[0],2,Rsh[0]))
        Wv = np.tensordot(L, ket, axes = [0,0])
        Wv = np.tensordot(Wv, MPOmat, axes = ([0,2],[0,2]))
        Wv = np.tensordot(Wv, R, axes = ([1,3],[0,1]))
        return np.reshape(Wv, (vec.shape))
    
    def double_site_matvec(self, vec: np.ndarray,
                           R: np.ndarray,
                           L: np.ndarray,
                           MPOi: np.ndarray, 
                           MPOi1: np.ndarray) -> np.ndarray:
        ''' Implements the double site matrix-vector multiplication for the 
            double site DMRG routine
            
        Args:
            vec: double site vector 'ket[i]-ket[i+1]' flattened to a vector to be multiplied by MPO
            R: right environment of 'ket[i+1]'
            L: left environment of 'ket[i]'
            MPOi: local MPO matrix to multiply against 'ket[i]'
            MPOi1: second local MPO matrix to multiply against 'ket[i+1]'
            
        Returns:
            Wv: vector which is the result of the multiplication of vec with the effective double
                site generator
        '''
        Rsh=R.shape
        Lsh=L.shape
        ket = np.reshape(vec, (Lsh[0],2,2,Rsh[0]))
        middle = np.tensordot(MPOi,MPOi1, axes = [3,0])
        Wv = np.tensordot(L, ket, axes = [0,0])
        Wv = np.tensordot(Wv, middle, axes = ([0,2,3],[0,2,4]))
        Wv = np.tensordot(Wv, R, axes = ([1,4],[0,1]))
        return np.reshape(Wv, (vec.shape))

    
    def optimize_single_site(self, site: int, 
                             sweep_direction: str,
                             tol: Optional[float]=1e-10,
                             Dmax: Optional[int] = None,
                             cutoff: Optional[float] = 0,
                             ncv: Optional[int] = None):
        ''' Implements a single site optimization. Updates the local environments and shifts 
            the mps.center position one site to the left or right, depending on 'sweep_direction'
        
        Args:
            site: the site to be updated
            sweep_direction: 'right' will move the mps.center position right and compute the left environment of 'site'
                             'left' will move the mps.center position left and compute the right environment of 'site'
            tol: Relative accuracy for eigenvalues (stopping criterion) 
            Dmax: maximal number of bond dimensions to keep per site
            cutoff: maximal absolute value for the singular values, SVs below this value are dropped
            ncv: The number of Lanczos vectors
            
        Returns:
            energy: leading local eigenvalue of the mpo after the optimization
            truncation_error: size of the total number of singular values discarded during the optimization
        '''
        sh=self.mps.tensors[site].shape
        vecshape = sh[0]*sh[1]*sh[2]
        bk = self.mps.tensors[site].reshape(vecshape)
        Weff = LinearOperator((vecshape,vecshape), 
                              matvec= lambda x: self.single_site_matvec(x,self.Rs[site],self.Ls[site],self.mpo.tensors[site]) )
        try:
            ev0, Rev = eigs(Weff,k=1,which='LR', v0=bk, tol=tol, ncv = ncv)
        except ArpackNoConvergence:
            print(f"\nAn exception occurred: eigenvector at site {site} did not converge")
            Rev = bk
            ev0 = np.array([1e100])
        Rev = Rev.real
        self.mps.tensors[site] = np.reshape(Rev, sh)
        
        if sweep_direction == 'right':
            truncation_error = self.mps.position(site+1, normalize_SVs = True,
                                                  Dmax = Dmax, cutoff = cutoff)
            self.add_L(site+1)            
        elif sweep_direction == 'left':
            truncation_error = self.mps.position(site-1, normalize_SVs = True, 
                                                  Dmax = Dmax, cutoff = cutoff)
            self.add_R(site-1)
        else:
            raise ValueError("Please specify sweep direction 'left' or 'right' for the update.")
        
        return ev0, truncation_error
    
    
    def single_site_sweep(self,
                          tol: Optional[float] = 1e-10,
                          Dmax: Optional[int] = None,
                          cutoff: Optional[float] = 0,
                          ncv: Optional[int] = None):
        ''' Implements a single site sweep of the DMRG algorithm. Sweeps from mps.center tot the right, 
            followed by a left sweep and again a right sweep back to mps.center.
        
        Args:
            tol: Relative accuracy for eigenvalues (stopping criterion) 
            Dmax: maximal number of bond dimensions to keep per site
            cutoff: maximal absolute value for the singular values, SVs below this value are dropped
            ncv: The number of Lanczos vectors
            
        Returns:
            energy: energy after the sweep
            truncation_error: size of the total number of singular values discarded during the sweep
        '''
        N=len(self.mps)
        truncation_error = 0
        site = self.mps.center

    #     Sweep to right-most node
        for i in range(site,N-1):
            ev0, err = self.optimize_single_site(i,'right',
                             tol = tol, Dmax = Dmax, cutoff = cutoff, ncv = ncv)           
            truncation_error += err
            
            stdout.write(f"\rVarMPS site={i}/{N}: "
                     f"optimized E={ev0[0].real}, D = {self.mps.tensors[i].shape[2]},   truncated SVs = {err} ")
            stdout.flush()
            
    #     Sweep to left-most node
        for i in range(N-1,0,-1):
            ev0, err = self.optimize_single_site(i,'left',
                             tol = tol, Dmax = Dmax, cutoff = cutoff, ncv = ncv)           
            truncation_error += err

            stdout.write(f"\rVarMPS site={i}/{N}: "
                     f"optimized E={ev0[0].real}, D = {self.mps.tensors[i].shape[0]},   truncated SVs = {err} ")
            stdout.flush()

        #Final update left-most node if site = 0, final right sweep up to site if site is non-zero 
        if site == 0:
            sh=self.mps.tensors[0].shape
            vecshape = sh[0]*sh[1]*sh[2]
            bk = self.mps.tensors[0].reshape(vecshape)
            Weff = LinearOperator((vecshape,vecshape), 
                                  matvec= lambda x: self.single_site_matvec(x,self.Rs[0],self.Ls[0],self.mpo.tensors[0]) )
            ev0, Rev = eigs(Weff,k=1,which='LR', v0=bk, tol=tol, ncv = ncv)
            Rev = Rev.real
            Rev = Rev/Rev.sum()
            self.mps.tensors[0] = np.reshape(Rev, sh)
            
            stdout.write(f"\rVarMPS site={0}/{N}: "
                     f"optimized E={ev0[0].real}, D = {self.mps.tensors[0].shape[0]},   truncated SVs = {0} ")
            stdout.flush()
            
        else:
            for i in range(0,site):
                ev0, err = self.optimize_single_site(i,'right',
                                 tol = tol, Dmax = Dmax, cutoff = cutoff, ncv = ncv)           
                truncation_error += err

                stdout.write(f"\rVarMPS site={i}/{N}: "
                         f"optimized E={ev0[0].real}, D = {self.mps.tensors[i].shape[2]},   truncated SVs = {err} ")
                stdout.flush()                

    #     if ev0.imag > 1e-14 :
    #         print("Imaginary eigenvalue: ", ev0.imag)

        return ev0[0].real, truncation_error
    
    def run_single_site_dmrg(self,
                             MaxSweeps: Optional[int] = 20,
                             accuracy: Optional[float] = 1e-8,
                             tol: Optional[float] = 1e-10,
                             Dmax: Optional[int] = None,
                             cutoff: Optional[float] = 0,
                             ncv: Optional[int] = None):
        ''' Runs the single site DMRG algorithm until convergence (or for 'MaxSweeps' number of sweeps)   
        
        Args:
            MaxSweeps: maximal number of sweeps
            accuracy: Global accuracy as convergence criteria for the variance in energy
            tol: Relative accuracy for local eigenvalues (stopping criterion) 
            Dmax: maximal number of bond dimensions to keep per site
            cutoff: maximal absolute value for the singular values, SVs below this value are dropped
            ncv: The number of Lanczos vectors
            
        Returns:
            energy: energy after the sweep
            variance: variance in energy of converged mps
            truncation_error: size of the total number of singular values discarded during the sweep
        '''
               
        N = len(self.mps)
        converged = False
        variance = 1
        num_sweeps = 0
        truncation_error = 0
        final_energy = 1e100
        
        if self.mpo.s is not None and self.mpo.s == 0:
            density_mpo = occupancy_MPO(N)
            
        self.compute_Rs()
        self.compute_Ls()
        start_time=time.time()
        
        while not converged:
            en, err = self.single_site_sweep(tol, Dmax, cutoff, ncv)
            truncation_error += err
            
            norm = self.mps.norm()
            if self.mpo.s == 0:
                energy = MPOexpectation(self.mps, density_mpo, 'stoch')/norm/N
            else:
                energy = MPOexpectation(self.mps, self.mpo, 'stoch')/norm
                variance = abs(MPOvariance(self.mps, self.mpo, 'stoch')/norm - energy**2)
            
            if variance < accuracy and self.mpo.s != 0 and num_sweeps>0:
                converged = True
            elif np.abs((final_energy - energy)/energy) < accuracy and self.mpo.s == 0:
                converged = True
            elif np.abs((final_energy - energy)/energy) < 1e-14:
                converged = True
            final_energy = energy
            num_sweeps += 1
            if num_sweeps >= MaxSweeps:
                print(f"\nMaxSweeps {num_sweeps} reached before convergence to desired accuracy")
                break
        
        if self.mpo.s == 0:
            variance = 1/N**2*MPOvariance(self.mps, density_mpo, 'stoch')/norm-final_energy**2
        
        end_time=time.time()
        compt = end_time - start_time
        
        print('\ns = %.6f,    n = %2i,    FE = %.9f,    delFE = %.9f    tps = %.2fs    <D>= %.2f   maxD = %i' %(self.mpo.s, num_sweeps, energy, variance, compt/num_sweeps, np.mean(self.mps.bond_dimensions[1:-1]), max(self.mps.bond_dimensions) ))
        
        return final_energy, variance, truncation_error
    
    def optimize_double_site(self, site: int, 
                             sweep_direction: str,
                             tol: Optional[float]=1e-10,
                             Dmax: Optional[int] = None,
                             cutoff: Optional[float] = 0,
                             ncv: Optional[int] = None):
        ''' Implements a double site optimization. Updates the local environments and shifts 
            the mps.center position one site to the left or right, depending on 'sweep_direction'
        
        Args:
            site: the sites to be updated will by convention be (site, site+1)
            sweep_direction: 'right' will move the mps.center position from site to site+1 
                              and compute the left environment of 'site+1'
                             'left' will move the mps.center position from site+1 to site and
                              compute the right environment of 'site'
            tol: Relative accuracy for eigenvalues (stopping criterion) 
            Dmax: maximal number of bond dimensions to keep per site
            cutoff: maximal absolute value for the singular values, SVs below this value are dropped
            ncv: The number of Lanczos vectors
            
        Returns:
            energy: leading local eigenvalue of the mpo after the optimization
            truncation_error: size of the total number of singular values discarded during the optimization
        '''
        twoket= np.tensordot(self.mps.tensors[site],self.mps.tensors[site+1], axes = [2,0])
        sh=twoket.shape
        vecshape = sh[0]*sh[1]*sh[2]*sh[3]
        bk = np.reshape(twoket,vecshape)
        Weff = LinearOperator((vecshape,vecshape), 
                              matvec= lambda x: self.double_site_matvec(x,
                                                                        self.Rs[site+1],
                                                                        self.Ls[site],
                                                                        self.mpo.tensors[site],
                                                                        self.mpo.tensors[site+1]) )
        try:
            ev0, Rev = eigs(Weff,k=1,which='LR', v0=bk, tol=tol, ncv = ncv)
        except ArpackNoConvergence :
            print(f"\nAn exception occurred: eigenvector at sites {site}, {site+1} did not converge")
            Rev = bk
            ev0 = np.array([1e100])
        Rev = Rev.real
        Rev = np.reshape(Rev, sh)
        if sweep_direction == 'right':
            self.mps.tensors[site], s, v, truncation_error = svd(Rev, 2, Dmax, cutoff)
            self.mps.tensors[site+1] = np.tensordot(np.diag(s), v, axes= [1,0]) 
            self.mps.center = site+1
            self.add_L(site+1)
        elif sweep_direction == 'left':
            u, s, self.mps.tensors[site+1], truncation_error = svd(Rev, 2, Dmax, cutoff)
            self.mps.tensors[site] = np.tensordot(u, np.diag(s), axes = [2,0])
            self.mps.center = site            
            self.add_R(site) 
        else:
            raise ValueError("Please specify sweep direction 'left' or 'right' for the update.")
        
        return ev0, truncation_error

        
    
    def double_site_sweep(self, 
                          tol: Optional[float] = 1e-10,
                          Dmax: Optional[int] = None,
                          cutoff: Optional[float] = 1e-16,
                          ncv: Optional[int] = None):
        ''' Implements a double site sweep of the DMRG algorithm
        
        Args:
            tol: Relative accuracy for eigenvalues (stopping criterion) 
            Dmax: maximal number of bond dimensions to keep per site
            cutoff: maximal absolute value for the singular values, SVs below this value are dropped
            ncv: The number of Lanczos vectors
            
        Returns:
            energy: energy after the sweep
            truncation_error: size of the total number of singular values discarded during the sweep
        '''
        N=len(self.mps)
        truncation_error = 0
#         ket = self.mps.tensors
#         MPO = self.mpo.tensors 
        site = self.mps.center

    #     Sweep to right-most odd node
        for i in range(site,N-1):
            ev0, err = self.optimize_double_site(i, 'right', tol, Dmax, cutoff, ncv)
            truncation_error += err
            
            stdout.write(f"\rVarMPS sites=({i},{i+1})/{N}: "
                         f"optimized E={ev0[0].real}, D = {self.mps.tensors[i].shape[2]},  truncated SVs = {err}  ")
            stdout.flush()

    #     Sweep to left-most node
        for i  in range(N-2,-1,-1):
            ev0, err = self.optimize_double_site(i, 'left', tol, Dmax, cutoff, ncv)
            truncation_error += err
            
            stdout.write(f"\rVarMPS sites=({i},{i+1})/{N}: "
                         f"optimized E={ev0[0].real}, D = {self.mps.tensors[i].shape[2]},  truncated SVs = {err}  ")
            stdout.flush()
            
        for i  in range(0,site):
            ev0, err = self.optimize_double_site(i, 'right', tol, Dmax, cutoff, ncv)
            truncation_error += err
            
            stdout.write(f"\rVarMPS sites=({i},{i+1})/{N}: "
                         f"optimized E={ev0[0].real}, D = {self.mps.tensors[i].shape[2]},  truncated SVs = {err}  ")
            stdout.flush()
            
        return ev0[0].real, truncation_error
    
    def run_double_site_dmrg(self,
                             MaxSweeps: Optional[int] = 20,
                             accuracy: Optional[float] = 1e-8,
                             tol: Optional[float] = 1e-10,
                             Dmax: Optional[int] = None,
                             cutoff: Optional[float] = 0,
                             ncv: Optional[int] = None):
        ''' Runs the double site DMRG algorithm until convergence (or for 'MaxSweeps' number of sweeps)
            Convergence is reached when the variance in the leading eigenvalue of the tilted generator is lower than 'accuracy', 
            unless tilting parameter s = 0, then the convergence criteria is based on the total density in the chain.
        
        Args:
            MaxSweeps: maximal number of sweeps
            accuracy: Global accuracy as convergence criteria for the variance in energy
            tol: Relative accuracy for local eigenvalues (stopping criterion) 
            Dmax: maximal number of bond dimensions to keep per site
            cutoff: maximal absolute value for the singular values, SVs below this value are dropped
            ncv: The number of Lanczos vectors
            
        Returns:
            energy: energy after the sweep
            variance: variance in energy of converged mps
            truncation_error: size of the total number of singular values discarded during the sweep
        '''
        N = len(self.mps)       
        converged = False
        variance = 1
        num_sweeps = 0
        truncation_error = 0
        if self.mpo.s == 0:
            density_mpo = occupancy_MPO(N)
            final_energy = MPOexpectation(self.mps, density_mpo, 'stoch')/self.mps.norm()/N
        else:
            final_energy = MPOexpectation(self.mps, self.mpo, 'stoch')/self.mps.norm()
            
        self.compute_Rs()
        self.compute_Ls()
        start_time=time.time()
        
        while not converged:
            en, err = self.double_site_sweep(tol, Dmax, cutoff, ncv)
            truncation_error += err
            
            norm = self.mps.norm()
            if self.mpo.s == 0:
                energy = MPOexpectation(self.mps, density_mpo, 'stoch')/norm/N
            else:
                energy = MPOexpectation(self.mps, self.mpo, 'stoch')/norm
                variance = abs(MPOvariance(self.mps, self.mpo, 'stoch')/norm - energy**2)
            
            if variance < accuracy and self.mpo.s != 0:
                converged = True
            elif np.abs((final_energy - energy)/energy) < accuracy and self.mpo.s == 0:
                converged = True
            elif np.abs((final_energy - energy)/energy) < 1e-12:
                converged = True
            final_energy = energy
            num_sweeps += 1
            if num_sweeps >= MaxSweeps:
                print(f"\nMaxSweeps {num_sweeps} reached before convergence to desired accuracy")
                break
        
        if self.mpo.s == 0:
            variance = 1/N**2*MPOvariance(self.mps, density_mpo, 'stoch')/norm-final_energy**2
            
        end_time=time.time()
        compt = end_time - start_time
        
        
        print('\ns = %.6f,    n = %2i,    FE = %.9f,    delFE = %.9f    tps = %.2fs    <D>= %.2f   maxD = %i' %(self.mpo.s, num_sweeps, final_energy, variance, compt/num_sweeps, np.mean(self.mps.bond_dimensions[1:-1]), max(self.mps.bond_dimensions) ))
        
        return final_energy, variance, truncation_error
