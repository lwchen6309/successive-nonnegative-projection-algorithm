import numpy as np

"""
A python implementation of successive nonnegative projection algorithm.

Reference:
"Successive Nonnegative Projection Algorithm for Robust Nonnegative Blind Source Separation" 
by Gillis. (2014), doi : 10.1137/130946782

"""

def simplexProj(y):
    """
    Given y,  computes its projection x* onto the simplex 
    
          Delta = { x | x >= 0 and sum(x) <= 1 }, 
    
    that is, x* = argmin_x ||x-y||_2  such that  x in Delta. 
    
    
    See Appendix A.1 in N. Gillis, Successive Nonnegative Projection 
    Algorithm for Robust Nonnegative Blind Source Separation, arXiv, 2013. 
    
    
    x = SimplexProj(y)
    
    ****** Input ******
    y    : input vector.
    
    ****** Output ******
    x    : projection of y onto Delta.
    """
    
    if len(y.shape) == 1: #Reshape to (1,-1) if y is a vector.
        y = y.reshape(1,-1)
    
    x = y.copy()
    x[x < 0] = 0
    K = np.flatnonzero(np.sum(x,0) > 1)
    x[:,K] = blockSimplexProj(y[:,K])
    return x

def blockSimplexProj(y):
    """ Same as function SimplexProj except that sum(max(Y,0)) > 1. """
    
    r, m = y.shape
    ys = -np.sort(-y,axis=0)
    mu = np.zeros(m, dtype=float)
    S = np.zeros((r,m), dtype=float)
    
    for i in range(1,r):
        S[i,:] = np.sum(ys[:i,:] - ys[i,:],0) 
        colInd_ge1 = np.flatnonzero(S[i,:] >= 1)
        colInd_lt1 = np.flatnonzero(S[i,:] < 1)
        if len(colInd_ge1) > 0:
            mu[colInd_ge1] = (1-S[i-1,colInd_ge1]) / i - ys[i-1,colInd_ge1]
        if i == r:
            mu[colInd_lt1] = (1-S[r,colInd_lt1]) / (r + 1) - ys[r,colInd_lt1]
    x = y + mu
    x[x < 0] = 0
    return x

def fastGrad_simplexProj(M, U, V=None, maxiter=500):
    """
    Fast gradient method to solve least squares on the unit simplex.  
    See Nesterov, Introductory Lectures on Convex Optimization: A Basic 
    Course, Kluwer Academic Publisher, 2004. 

    This code solves: 

                min_{V(:,j) in Delta, forall j}  ||M-UV||_F^2, 

    where Delta = { x | sum x_i <= 1, x_i >= 0 for all i }.
     
    See also Appendix A in N. Gillis, Successive Nonnegative Projection 
    Algorithm for Robust Nonnegative Blind Source Separation, arXiv, 2013. 


    [V,e] = FGMfcnls(M,U,V,maxiter) 

    ****** Input ******
    M      : m-by-n data matrix
    U      : m-by-r basis matrix
    V      : initialization for the fast gradient method 
             (optional, use [] if none)
    maxiter: maximum numbre of iterations (default = 500). 

    ****** Output ******
    V      : V(:,j) = argmin_{x in Delta}  ||M-Ux||_F^2 forall j. 
    e      : e(i) = error at the ith iteration
    """
    
    m, n = M.shape
    m, r = U.shape
    
    # Initialization of V
    if V is None:
        V = np.zeros((r,n),dtype=float) 
        for col_M in range(n):
            # Distance between ith column of M and columns of U
            disti = np.sum((U - M[:,col_M].reshape(-1,1))**2, 0)
            min_col_U = np.argmin(disti)
            V[min_col_U, col_M] = 1
    
    # Hessian and Lipschitz constant
    UtU = U.T.dot(U)
    L = np.linalg.norm(UtU,ord=2) # 2-norm
    # Linear term 
    UtM = U.T.dot(M)
    nM = np.linalg.norm(M)**2 # Frobenius norm
    # Projection
    alpha = [0.05, 0] # Parameter, can be tuned. 
    err = [0, 0]
    V = simplexProj(V) # Project initialization onto the simplex
    Y = V # second sequence
    
    delta = 1e-6
    # Stop if ||V^{k}-V^{k+1}||_F <= delta * ||V^{0}-V^{1}||_F
    for i in range(maxiter):
        # Previous iterate
        Vprev = V
        # FGM Coefficients
        alpha[1] = (np.sqrt(alpha[0]**4 + 4*alpha[0]**2) - alpha[0]**2) / 2
        beta = alpha[0] * (1 - alpha[0]) / (alpha[0]**2 + alpha[1])
        # Projected gradient step from Y
        V = simplexProj(Y - (UtU.dot(Y) - UtM) / L)
        # `Optimal' linear combination of iterates
        Y = V + beta * (V - Vprev)
        # Error
        err[1] = nM - 2 * np.sum(np.ravel(V*UtM)) + np.sum(np.ravel(UtU * V.dot(V.T)))
        
        # Restart: fast gradient methods do not guarantee the objective
        # function to decrease, a good heursitic seems to restart whenever it
        # increases although the global convergence rate is lost! This could
        # be commented out. 
        if i > 0 and err[1] > err[0]:
            Y = V
        if i is 0:
            eps0 = np.linalg.norm(V - Vprev)
        eps = np.linalg.norm(V - Vprev)
        if eps < delta * eps0:
            break
        # Update
        alpha[0] = alpha[1]
        err[0] = err[1]
        
    return V, err[1]
    
def snpa(M, r, normalize=False, maxitn=100):
    """
    Successive Nonnegative Projection Algorithm (variant with f(.) = ||.||^2)
    
    *** Description ***
    At each step of the algorithm, the column of M maximizing ||.||_2 is 
    extracted, and M is updated with the residual of the projection of its 
    columns onto the convex hull of the columns extracted so far. 
    
    See N. Gillis, Successive Nonnegative Projection Algorithm for Robust 
    Nonnegative Blind Source Separation, arXiv, 2013. 
     
    
    [J,H] = SNPA(M,r,normalize) 
    
    ****** Input ******
    M = WH + N : a (normalized) noisy separable matrix, that is, W is full rank, 
                 H = [I,H']P where I is the identity matrix, H'>= 0 and its 
                 columns sum to at most one, P is a permutation matrix, and
                 N is sufficiently small. 
    r          : number of columns to be extracted. 
    normalize  : normalize=1 will scale the columns of M so that they sum to one,
                 hence matrix H will satisfy the assumption above for any
                 nonnegative separable matrix M. 
                 normalize=0 is the default value for which no scaling is
                 performed. For example, in hyperspectral imaging, this 
                 assumption is already satisfied and normalization is not
                 necessary. 
    
    ****** Output ******
    J        : index set of the extracted columns. 
    H        : optimal weights, that is, H argmin_{X >= 0} ||M-M(:,K)X||_F
    """
    
    m, n = M.shape
    
    if normalize:
        # Normalization of the columns of M so that they sum to one
        M /= (np.sum(M, 0) + 1e-15)
    
    normM = np.sum(M**2, 0)
    nM = np.max(normM)
    J = np.array([],dtype=int)
    # Perform r recursion steps (unless the relative approximation error is 
    # smaller than 10^-9)
    for i in range(r):
        if np.max(normM) / nM <= 1e-9:
            break
            
        # Select the column of M with largest l2-norm
        b = np.argmax(normM)
        a = normM[b]
        
        # Norms of the columns of the input matrix M 
        if i is 0:
            normM1 = normM.copy()
        
        # Check ties up to 1e-6 precision
        b = np.flatnonzero((a - normM) / a <= 1e-6)
        
        # In case of a tie, select column with largest norm of the input matrix M 
        if len(b) > 1:
            d = np.argmax(normM1[b])
            b = b[d]
        # Update the index set, and extracted column
        J = np.append(J,int(b))
        
        # Update residual 
        if i is 0:
            # Initialization using 10 iterations of coordinate descent
            # H = nnlsHALSupdt(M,M(:,J),[],10); 
            # Fast gradient method for min_{y in Delta} ||M(:,i)-M(:,J)y||
            H, _ = fastGrad_simplexProj(M,M[:,J],None,maxitn)
        else:
            H[:,J[i]] = 0
            h = np.zeros((1,n),dtype=float)
            h[0,J[i]] = 1
            H = np.vstack([H,h])
            H, _ = fastGrad_simplexProj(M,M[:,J],H,maxitn)
            
        # Update norms
        R = M - M[:,J].dot(H)
        normM = np.sum(R**2, 0)
        
    return J, H
    
import unittest
class TestSimplexProj(unittest.TestCase):
    def test_blockSimplexProj(self):
        y = np.arange(-3,5).reshape(2,-1)
        x = blockSimplexProj(y)
        x_answer = np.array([[0, 0, 0, 0], 
                            [1, 1, 1, 1]])
        np.testing.assert_array_equal(x, x_answer)
        
    def test_simplexProj(self):
        y = np.arange(-3,5).reshape(2,-1)
        x = simplexProj(y)
        x_answer = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])    
        np.testing.assert_array_equal(x, x_answer)
    
    def test_fastGrad_simplexProj(self):
        # M = np.array([
             # [8, 1, 6],
             # [3, 5, 7],
             # [4, 9, 2]]) / 10
        # x = np.ones(3,dtype=float).reshape(-1,1)/3
        # y = M.dot(x)
        # simplex_x, err = fastGrad_simplexProj(M,y)
        # simplex_x_answer = np.ones((1,3),dtype=float)
        # np.testing.assert_allclose(simplex_x, simplex_x_answer, atol=1e-10)
        
        import scipy.io as sio
        fgm_debug = sio.loadmat('./data/FGM_debug.mat')
        H = fgm_debug['H']
        H_ans = fgm_debug['H_ans']
        M = fgm_debug['M']
        J = fgm_debug['J'].ravel() - 1
        H_my, _ = fastGrad_simplexProj(M, M[:,J], H, 100)
        np.testing.assert_allclose(H_my, H_ans, atol=1e-10)
        
    def test_snpa(self):
        M = np.array([
             [8, 1, 6],
             [3, 5, 7],
             [4, 9, 2]])
        J, H = snpa(M,3)
        J_answer = np.array([1,0,2],dtype=int)
        H_answer = np.array([[0,1,0],[1,0,0],[0,0,1]], dtype=float)
        np.testing.assert_array_equal(J, J_answer)
        np.testing.assert_allclose(H, H_answer, atol=1e-10)
    
    
if __name__ == '__main__':
    unittest.main()

