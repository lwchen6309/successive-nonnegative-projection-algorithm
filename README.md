# Near-separable-NMF-on-time-resolved-spectra

An MATLAB implementaion of  
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


## (3) Algorithm
(a) A time-resolved sepctrum M is factorized on near-separable assumption to **get a nice initial guess of W and H**,  
by using the **successive nonnegative projection algorithm**

 * "Successive Nonnegative Projection Algorithm for Robust Nonnegative Blind Source Separation"  
    by Gillis. (2014), doi : 10.1137/130946782  
    code : https://sites.google.com/site/nicolasgillis/code  

(b) then perform a normal nmf (without near-separable assupmtion) to **improve the reconstruction accuracy**.  
In this work, a **projected gradient methods** is used.

 * "Projected gradient methods for non-negative matrix factorization"  
    by C.-J. Lin (2007), doi : 10.1162/neco.2007.19.10.2756  
    code : https://www.csie.ntu.edu.tw/~cjlin/nmf/index.html  

**The above nmf codes are directly used without modification, you can also download them from their authors' website.**

# Program
    "trs_fm.m" perform the separable nonnegative matrix factorization method.

    "nmf/*" is the code of nmf by using projected gradient methods.

    "snpa/*" is the code of nmf by using successive nonnegative projection algorithm.

    Run "nmf_demo.m" to generate time-resolved spectra, and perform the "trs_fm".

# Results

An artificial time-resolved spectra is given M = W * H:  
where  
## (1) The original time-resolved spectra, kinetics and hyperspectra

### M (time-resolved spectra)  
  
![image](https://github.com/j6309355065/Near-seperable-NMF-on-time-resolved-spectra/blob/master/figure/trspec.jpg)
  
### W (kinetics)  
  
![image](https://github.com/j6309355065/Near-seperable-NMF-on-time-resolved-spectra/blob/master/figure/trace.jpg)
  
### H (hyperspectra)  
  
![image](https://github.com/j6309355065/Near-seperable-NMF-on-time-resolved-spectra/blob/master/figure/spectra.jpg)
  
## (2) The results of separable NMF and the comparison with the originals  

### W and W_recontructed (kinetics)  
  
![image](https://github.com/j6309355065/Near-seperable-NMF-on-time-resolved-spectra/blob/master/figure/nmf_trace.jpg)
  
### H and H_recontructed (hyperspectra)  
  
![image](https://github.com/j6309355065/Near-seperable-NMF-on-time-resolved-spectra/blob/master/figure/nmf_spectra.jpg)
