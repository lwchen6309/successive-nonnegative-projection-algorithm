# Near-separable-NMF-on-time-resolved-spectra by Successive Nonnegative Projection Algorithm

An python implementaion of  

**"Successive Nonnegative Projection Algorithm for Robust Nonnegative Blind Source Separation"  
    by Gillis. (2014), doi : 10.1137/130946782  
    matlab code : https://sites.google.com/site/nicolasgillis/code  

**"Using Separable Nonnegative Matrix Factorization Techniques for the Analysis of Time-Resolved Raman Spectra"  
by Luce et al. (2016), doi : 10.1177/0003702816662600**


# Introduction
## (1) Time-resolved sepctrum
For fundamental knowledge of time-resolved spectrum, refer :  
https://en.wikipedia.org/wiki/Time-resolved_spectroscopy

Given a time-resolved sepctrum M ∈ Rm×n+, (m : time slices, n : pixels)  
we aimed to factorized M = W * H, where  
W ∈ Rm×r+ is the temporal profiles of the species involved in the reaction, (kinetics)  
and H ∈ Rr×n+ is their hyperspectra. (ex : absorption spectra, emission spectrum, fluorescence spectra, etc.)  
The decomposition can be done by the nonnegative matrix factorization techniques.  

Since the general NMF problem is a difficult problem (ill-posed, NP hard),  
it is more practical to solve the "separable NMF" problem, which gives an approximated solution.  
**For details of NMF and separable-NMF, refer the Luce's work linked above.**


## (2) Near-separable matrix
A matrix M is r-separable (near-separable) if there exist an index set K of cardinality r and a nonnegative matrix H ∈ Rr×n+ 
with M = M(:,K) H. That is, M can be spanned by a convex hull formed by M(:,K).  

**If you are confused by the idea of convex hull, imaging a r-dimensional space D ∈ Rr, where each axes correspond to M(:,K). 
The nonnegative H indicates that all elements of M has to live in the subspace D ∈ Rr+, which results a convex hull.**
ex : for r = 3, the covex hull is the first quadrant.

A time-resolved sepctrum M is near-separable, if 
**there exist a certain pixel that has only one hyperspectrum for all time slices.**  
Hence, the approximation of separable NMF highly depends on the overlapping between hyperspectra.  
**i.e. The less hyperspectra overlap, the better approximation you get.**


# Run program
    (1) Install libraries : numpy, scipy, matplotlib
    
    (2) Run time_resolved_spectra_NMF.py 

# Results

An artificial time-resolved spectra is given M = W * H:  
where  
### M (time-resolved spectra)  
![image](https://github.com/lwchen6309/successive-nonnegative-projection-algorithm./blob/master/image/trspec.png)

## The result of SNPA
from top to bottom: 
(1) H (hyperspectra from SNPA).
(2) spectra at different time slices.
(3) W(kinetics) and comparison of real ideal(real)-trace. 

![image](https://github.com/lwchen6309/successive-nonnegative-projection-algorithm./blob/master/image/NMF_result.png)
  
