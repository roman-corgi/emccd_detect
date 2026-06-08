# -*- coding: utf-8 -*-
"""Generate random numbers according to EM gain pdfs."""

import numpy as np
from scipy import special
from scipy.special import gammaln
try:
    from emccd_detect.partial_CIC_MLE import _LogPn, _LogGamma
except:
    pass
try:
    from partial_CIC_MLE import _LogPn, _LogGamma
except:
    pass
try:
    from joblib import Parallel, delayed
    import multiprocessing
except:
    pass
import scipy.linalg as la

from numba import njit, prange

import numpy as np
from numba import njit, prange
import time

@njit(parallel=True, cache=True)
def _numba_square_upper_triangular(M):
    """
    Parallelized upper-triangular matrix squaring.
    Only computes elements where col >= row, cutting standard multiplications in half.
    """
    m = M.shape[0]
    res = np.zeros((m, m), dtype=M.dtype)
    
    # Parallelize the rows across available CPU cores
    for r in prange(m):
        for c in range(r, m):
            total = 0.0
            # Inner loop only traverses the valid overlapping non-zero range
            for k in range(r, c + 1):
                total += M[r, k] * M[k, c]
            res[r, c] = total
            
    return res

@njit(cache=True)
def _numba_vector_matrix_mult(vec, M):
    """
    Optimized Vector x Upper-Triangular Matrix multiplication.
    Skips operations where the matrix elements are guaranteed to be zero.
    """
    m = M.shape[0]
    res = np.zeros(m, dtype=M.dtype)
    
    for c in range(m):
        total = 0.0
        # M[r, c] is only non-zero when r <= c
        for r in range(c + 1):
            total += vec[r] * M[r, c]
        res[c] = total
        
    return res

@njit(cache=True)
def fast_numba_triangular_power(A, row_idx, p):
    """
    Computes a specific row of (A^p) where A is an upper triangular matrix.
    Fully compiled in Numba for maximum execution speed.
    
    Parameters:
        A (NumPy.ndarray): An n x n upper triangular matrix (float64 or float32).
        row_idx (int): The 0-indexed row number to compute.
        p (int): The large exponent power (must be >= 0).
    """
    n = A.shape[0]
    
    # Base Case: Identity row if power is 0
    if p == 0:
        identity_row = np.zeros(n, dtype=A.dtype)
        identity_row[row_idx] = 1.0
        return identity_row

    # Step 1: Truncate the matrix to only the lower-right submatrix.
    # We copy it so we don't accidentally overwrite or mutate your original matrix.
    U = A[row_idx:n, row_idx:n].copy()
    m = U.shape[0]
    
    # Step 2: Initialize the row accumulator.
    # In the truncated space, our target row is always index 0.
    row_vec = np.zeros(m, dtype=A.dtype)
    row_vec[0] = 1.0
    
    # Step 3: Binary Exponentiation (Square-and-Multiply) Loop
    # This loop runs entirely in native machine code.
    while p > 0:
        if p % 2 == 1:
            row_vec = _numba_vector_matrix_mult(row_vec, U)
            #row_vec = np.dot(row_vec, U) # relies on BLAS, which is already optimized and multi-threaded in NumPy
        
        # Only square U if we actually need it for the next loop iterations
        if p > 1:
            U = _numba_square_upper_triangular(U)
            
        p //= 2

    # Step 4: Map the truncated result vector back to full n-dimensional space
    full_row = np.zeros(n, dtype=A.dtype)
    full_row[row_idx:] = row_vec
    
    return full_row



def fast_upper_triangular_row_power(A, row_idx, p):
    """
    Computes a specific row of (A^p) where A is an upper triangular matrix.
    
    Parameters:
        A (numPy.ndarray): An n x n upper triangular matrix.
        row_idx (int): The 0-indexed row number to compute.
        p (int): The large exponent power (must be >= 0).
    """
    n = A.shape[0]
    if p == 0:
        identity_row = np.zeros(n, dtype=A.dtype)
        identity_row[row_idx] = 1
        return identity_row

    # Step 1: Truncate the matrix. We only need the lower-right submatrix.
    # It starts at (row_idx, row_idx) and goes to the bottom right corner (n, n).
    U = A[row_idx:n, row_idx:n].copy()
    m = U.shape[0]
    
    # Step 2: Initialize the target row vector.
    # In the truncated space, our target row is the very first row (index 0).
    # We initialize it as the identity row to accumulate the power correctly.
    row_vec = np.zeros(m, dtype=A.dtype)
    row_vec[0] = 1
    
    # Step 3: Custom Upper Triangular Matrix Squaring function
    def square_upper_triangular(M):
        # standard np.dot(M, M) computes zeros below the diagonal.
        # For maximum speed in pure Python/NumPy, we can use a masked or 
        # triangular-aware approach, or rely on BLAS via np.dot which is heavily 
        # optimized. However, enforcing the upper-triangular structure explicitly
        # eliminates numerical drift and saves memory.
        # (For large matrices, BLAS np.dot is faster than a pure Python loop, 
        # so we clear out the lower triangle manually to preserve structure)
        res = np.dot(M, M)
        return np.triu(res)

    # Step 4: Binary Exponentiation (Square-and-Multiply) Loop
    while p > 0:
        if p % 2 == 1:
            # Vector-Matrix multiplication: row_vec = row_vec @ U
            # Because U is upper triangular, we can optimize this slightly,
            # but a 1D @ 2D dot product in NumPy is already highly optimized.
            row_vec = np.dot(row_vec, U)
        
        U = square_upper_triangular(U)
        p //= 2


    # Step 5: Map the truncated row vector back to the full n-dimensional row
    full_row = np.zeros(n, dtype=A.dtype)
    full_row[row_idx:] = row_vec
    
    return full_row

def get_eigenvectors_from_eigenvalues(A, eigenvalues, max_iter=2, tol=1e-10):
    n = A.shape[0]
    eigenvectors = np.zeros((n, len(eigenvalues)))
    identity = np.eye(n)
    
    for i, lam in enumerate(eigenvalues):
        # 1. Shift the matrix: (A - λI)
        shifted_A = A - lam * identity
        
        # 2. Add a tiny perturbation if λ is an exact eigenvalue to prevent exact singularity
        shifted_A += identity * 1e-12 
        
        # 3. Compute LU Decomposition once for this eigenvalue: O(n^3)
        lu, piv = la.lu_factor(shifted_A)
        
        # 4. Start with a random initial guess vector
        x = np.random.rand(n)
        x /= la.norm(x)
        
        # 5. Inverse iteration loop: O(n^2) per step
        for _ in range(max_iter):
            # Solve (A - λI) x_new = x
            x_new = la.lu_solve((lu, piv), x)
            
            norm_x_new = la.norm(x_new)
            if norm_x_new < tol:
                break
                
            x_new /= norm_x_new
            
            # Check for convergence early if needed
            if la.norm(x_new - x) < tol or la.norm(x_new + x) < tol:
                x = x_new
                break
            x = x_new
            
        eigenvectors[:, i] = x
        
    return eigenvectors

class RandEMGainException(Exception):
    """Exception class for rand_em_gain module."""


def rand_em_gain(n_in_array, em_gain, numel_gain_register, threshold):
    """Generate random numbers according to EM gain pdfs.

    Parameters
    ----------
    n_in_array : array_like
        Array of electron values (e-).
    em_gain : float
        EM gain multiplication factor.
    threshold : float
        Threshold for switching between methods.  If the product of n_in and 
        the size of the array is greater than this threshold, the faster, less memory-intensive method 
        (gamma distribution) is used.  Otherwise, the more accurate method (Pn) is used.

    Returns
    -------
    array_like
        Electron values multiplied by random EM gain distribution (e-).

    Notes
    -----
    This function returns an array of the same size as n_in_array. Every element
    in n_in_array is multiplied by em_gain*rand_val, where rand_val is a random
    number drawn from a specific pdf selected based on the value of the
    n_in_array element.

    References
    ----------
    [1] http://matlabtricks.com/post-44/generate-random-numbers-with-a-given-distribution
    [2] https://arxiv.org/pdf/astro-ph/0307305.pdf
    [3] https://arxiv.org/abs/2405.17622

    """
    if em_gain < 1:
        raise RandEMGainException('EM gain cannot be set to less than 1')
    elif em_gain == 1:
        return n_in_array
    else:
        n_out_array = _apply_gain(n_in_array, em_gain, numel_gain_register, threshold)
        return n_out_array

def _apply_gain(n_in_array, em_gain, numel_gain_register, threshold):
    """Apply a specific em_gain to all nonzero n_in values."""
    # Initialize output count array
    n_out_array = np.zeros_like(n_in_array)

    # need integer values for Pn().  Otherwise, we try to keep e- values as floats since EM gain,
    # k gain, nonlinearity, master flat, etc are calibrated assuming fractions of electrons, and 
    # we can get fractions of electrons here for particular gain values.  So we just round 
    # DN output to integer at the end.
    n_in_array = np.round(n_in_array) 

    gamma_inds = np.where(n_in_array*(n_in_array*em_gain - n_in_array) >= threshold) #XXX change?
    not_gamma_inds = np.where(n_in_array*(n_in_array*em_gain - n_in_array) < threshold)
    n_out_array[gamma_inds] = np.random.gamma(n_in_array[gamma_inds], em_gain)
    # For the others, get unique nonzero n_in values
    n_in_unique = np.unique(n_in_array[not_gamma_inds])
    
    if n_in_unique.size != 0:
        n_in_unique = n_in_unique[n_in_unique > 0]
        # Generate random numbers according to the gain distribution for each n_in
        for n_in in n_in_unique:                
            inds = np.where(n_in_array == n_in)[0]
            n_out_array[inds] = _rand_pdf(int(np.round(n_in)), em_gain, len(inds), numel_gain_register, threshold=threshold) 

    return np.round(n_out_array)

def _rand_pdf(n_in, em_gain, size, numel_gain_register, threshold=1e7):
    """Draw samples from the EM gain distribution assuming Matsuo's exact 
    PDF (via Brian Sutin's formulation).
    
    Brian Sutin's paper: 
    https://doi.org/10.1117/1.JATIS.9.2.028001 
    """
    y = np.random.random(size) # ranges from [0,1)

    n_out = np.ones_like(y).astype(float)*np.nan
    # as a good initial guess, generate up to n_in*em_gain*y/0.5, which is the Erlang mean scaled by y relative to 0.5 prob, where the mean is 
    x = np.arange(n_in, max(np.round(n_in*em_gain*y.max()/0.5), n_in+10))
    if n_in * x.size >= threshold: #was 2e8 originally 
        #revert back to gamma distribution, valid for large x values. 
        # When n*g is large, the mean will be large, and most of the non-zero CDF
        # happens around there, but if a small y value is requested, it could 
        # be that the corresponding x is small, so this condition is a little 
        # better than using gamma distribution when n*g is large. We consider
        # n_in*x.size b/c that is the practical memory bottleneck which can stop the program.
        n_out = np.random.gamma(n_in, em_gain, size=size) # then the while loop will be skipped, and this n_out will be returned
    cdfsum = np.array([0]) #initialize 
    counter = 0
    while np.where(np.isnan(n_out))[0].size > 0: 
        cdfsum = np.append(cdfsum[-1], cdfsum[-1] + np.cumsum(exact_gain_PDF(em_gain, numel_gain_register, n_in, x))) 
        x = np.append(x[0]-1, x) # add one more term to sum to match cdfsum
        # find where y values should go in cdfsum array
        keep_inds = np.where(y <= cdfsum[-1])
        keep_inds = np.intersect1d(keep_inds, np.where(y >= cdfsum[0]))
        cdf_inds = np.searchsorted(cdfsum.data, y[keep_inds])
        cdf_inds2 = np.where(cdf_inds == 0, 1, cdf_inds) # to avoid out of bounds error when cdf_inds is 0; in this case where 0 is in cdf_inds, both options below are the same
        # if 1, x[cdf_inds] closer to y
        preferred_x = np.less(np.abs(cdfsum[cdf_inds] - y[keep_inds]), np.abs(cdfsum[cdf_inds2-1] - y[keep_inds])).astype(int)
        # if 1, x[cdf_inds-1] closer to y
        preferred_x_1 = 1 - preferred_x
        n_out[keep_inds] = preferred_x_1 * x[cdf_inds2-1] + preferred_x * x[cdf_inds]
        if counter == 0:
            n_out[np.where(y < cdfsum[0])] = n_in 
        x = np.arange(x[-1]+1, 2*(x[-1]+1)) # increase x range if needed 
        counter += 1

    return np.round(n_out)

def _rand_pdf_Pn(n_in, em_gain, size, threshold=1e7):
    """Draw samples from the EM gain distribution.
    
    paper reference regarding Pn:
    https://doi.org/10.1117/1.JATIS.11.1.018005
    """
    y = np.random.random(size) # ranges from [0,1)

    # Use exact solutions for n_in == 1; no exact solution for n_in >= 2 (when summing PDF to get CDF; there is 
    # exact solution for n_in=2 if you integrate PDF to get CDF) 
    if n_in == 1:
        n_out = -em_gain * np.log(1 - y)
    else:
        # For n > 1, sum PDF to get CDF. 
        n_out = np.ones_like(y).astype(float)*np.nan
        # as a good initial guess, generate up to n_in*em_gain*y/0.5, which is the Erlang mean scaled by y relative to 0.5 prob, where the mean is 
        x = np.arange(n_in, max(np.round(n_in*em_gain*y.max()/0.5), n_in+10))
        if n_in * x.size >= threshold: #was 2e8 originally 
            #revert back to gamma distribution, valid for large x values. 
            # When n*g is large, the mean will be large, and most of the non-zero CDF
            # happens around there, but if a small y value is requested, it could 
            # be that the corresponding x is small, so this condition is a little 
            # better than using gamma distribution when n*g is large. We consider
            # n_in*x.size b/c that is the practical memory bottleneck which can stop the program.
            n_out = np.random.gamma(n_in, em_gain, size=size) # then the while loop will be skipped, and this n_out will be returned
        cdfsum = np.array([0]) #initialize 
        counter = 0
        while np.where(np.isnan(n_out))[0].size > 0: 
            cdfsum = np.append(cdfsum[-1], cdfsum[-1] + np.cumsum(Pn(n_in, em_gain, x)))
            x = np.append(x[0]-1, x) # add one more term to sum to match cdfsum
            # find where y values should go in cdfsum array
            keep_inds = np.where(y <= cdfsum[-1])
            keep_inds = np.intersect1d(keep_inds, np.where(y >= cdfsum[0]))
            cdf_inds = np.searchsorted(cdfsum.data, y[keep_inds])
            cdf_inds2 = np.where(cdf_inds == 0, 1, cdf_inds) # to avoid out of bounds error when cdf_inds is 0; in this case where 0 is in cdf_inds, both options below are the same
            # if 1, x[cdf_inds] closer to y
            preferred_x = np.less(np.abs(cdfsum[cdf_inds] - y[keep_inds]), np.abs(cdfsum[cdf_inds2-1] - y[keep_inds])).astype(int)
            # if 1, x[cdf_inds-1] closer to y
            preferred_x_1 = 1 - preferred_x
            n_out[keep_inds] = preferred_x_1 * x[cdf_inds2-1] + preferred_x * x[cdf_inds]
            if counter == 0:
                n_out[np.where(y < cdfsum[0])] = n_in 
            x = np.arange(x[-1]+1, 2*(x[-1]+1)) # increase x range if needed 
            counter += 1

    return np.round(n_out)

def CDFErlang_root(x, n, g, y):
    """We use the cumulative distribution function (CDF) for the EM gain 
    probability distribution function (PDF), approximate but appropriate for 
    large EM gain.  This function is used to find the x value when the CDF is 
    equal to y.  The PDF is the Erlang distribution (gamma distribution for 
    integer-valued n), which is normalized when it is integrated.  However, 
    since x should also be integer-valued, the sum should be used instead.  
    In that case, there is a normalization factor dependent on g and n, but the 
    factor is very close to 1 in general.  Since we are already approximating
    by using this PDF, we leave off this troublesome factor, which is in terms
    of the Hurwitz-Lerch transcendent. 
    
    Parameters
    ----------
    x : array-like (1D)
        PDF variate values, one for each value in n array.
    n : array-like (1D)
        Input array of electron values (e-) before entering the gain register.
    g: float
        EM gain value.
    y : float
        Desired CDF value.

    Returns
    -------
    array-like (1D)
        CDF values.
    """
    CDF = 1 - special.gammaincc(n, x/g)
    return CDF - y

def Pn(n, g, x):
    """The probability distribution function (PDF) for the normalized EM gain, 
    valid for small and large gain values.
    
    Parameters
    ----------
    n : array-like (1D)
        Input array of electron values (e-) before entering the gain register.
    g: float
        EM gain value.
    x : array-like (1D)
        PDF variate values, one for each value in n array.

    Returns
    -------
    array-like (1D)
        PDF values.

    """
    out = _LogPn(n, g, x)
    Pn = np.exp(out)
    return Pn

def exact_gain_PDF(gain, M, w, x):
    '''Using Brian Sutin's formulation of Matsuo, 
    a form much easier to compute.

    Brian's paper:
    https://doi.org/10.1117/1.JATIS.9.2.028001
    '''
    nmax = int(x.max())
    P0 = np.zeros(nmax+1)
    P0[w] = 1 # w is the number of electrons incident on the gain register
    B = np.zeros((nmax+1, nmax+1))
    Q = gain**(1/M) -1 #Q is the probability of multiplication in a given stage
    if Q >= 1:
        raise ValueError('Gain and M values are not compatible. Q must be <1.')
    n, k = np.meshgrid(np.arange(nmax+1), np.arange(nmax+1))
    B[k,n] = np.exp(gammaln(k+1) - gammaln(n-k+1) - gammaln(2*k-n+1) + (2*k-n)* np.log(1-Q) + (n-k)*np.log(Q))
    inds=np.where(k>n)
    B[inds] = 0 

    # eigenvalues = np.diag(B)
    # eigenvectors = get_eigenvectors_from_eigenvalues(B, eigenvalues)

    # Compute eigenvalues (D) and eigenvectors (V)
    # eigenvalues, eigenvectors = np.linalg.eig(B)
    # Raise the diagonal entries to the power of n
    # D_power_n = np.diag(eigenvalues ** M)
    # # Reconstruct the final matrix: V * (D^n) * V^-1
    # V_inv = np.linalg.inv(eigenvectors)
    # Bpower = eigenvectors @ D_power_n @ V_inv

    # matrix is traingular, so sparse.  For speed and less memory usage:
    # sparse_B = sparse.csc_array(B)
    # # eigenvalues, eigenvectors = sparse.linalg.eigs(sparse_B, k=7)
    # # D_power_n = np.diag(eigenvalues ** M)
    # # # Reconstruct the final matrix: V * (D^n) * V^-1
    # # V_inv = sparse.linalg.inv(eigenvectors)
    # # Bpower = eigenvectors @ D_power_n @ V_inv
    # Bpower = sparse.linalg.matrix_power(sparse_B, M)
    # P = P0.dot(Bpower)

    #t = time.time()
    P = fast_upper_triangular_row_power(B, w, M)

    #P = fast_numba_triangular_power(B, w, M)
    #print('Time for fast_numba_triangular_power:', time.time() - t)
    return P[x.astype(int)]

def partial_CIC(array_size, em_gain, numel_gain_register, gain_CIC_Q, 
                gain_CIC_specs, threshold):
    '''Computes the contribution of partial CIC (clock-induced charge generated 
    in the gain register).  

    reference to paper regarding partial CIC:
    https://doi.org/10.1117/1.JATIS.11.1.018005
    
    Parameters
    ----------
    array_size : int
        Number of pixels.
    em_gain : float
        EM gain of full gain register.
    numel_gain_register : int
        Number of elements/stages in EM gain register.
    gain_CIC_Q : float 
        Probability Q (or mean rate) of production of a clock-induced charge (CIC)
        in a given gain register stage. the gain register. We call this "partial CIC". 
    gain_CIC_specs : dict
        This input supercedes gain_CIC_Q and renders the value of gain_CIC_Q 
        irrelevant.  This is used for specifying particular "hot" stages which source the 
        CIC produced in the gain register.  If None, gain_CIC_Q assumed for all
        gain register stages. If a dictionary is provided, the keys should be 
        integer-valued and be the number of stages until the end (e.g., 1 means 
        CIC appears in the last stage and gets clocked through that 1 gain stage),
        and the values for the dictionary should be the corresponding Q values. 
    threshold : float
        Threshold for switching between methods.  If the product of n_in and 
        the size of the array is greater than this threshold, the faster, less memory-intensive method 
        (gamma distribution) is used.  Otherwise, the more accurate method (Pn) is used.

    Returns
    -------
    array-like (1D)
        partial CIC contribution

    '''
    partial_CIC = np.zeros(array_size).astype(float)
    # probability of multiplication in gain register for a given stage
    P = em_gain**(1/numel_gain_register) - 1

    # CIC events in gain register: Poisson
    # Each pixel goes through all numel_gain_register stages
    if gain_CIC_specs is None:
        CIC_stages = np.random.poisson(gain_CIC_Q, size=array_size*numel_gain_register)
        # loop through, numel_gain_register entries at a time
        # Each of the numel_gain_register stage_chunks has length array_size
        stage_chunks = np.array_split(CIC_stages, numel_gain_register)
        stage_numbers = np.arange(1, numel_gain_register) #leaves off the entrance stage since regular CIC accounts for n_in there
    else:
        stage_chunks = []
        stage_numbers = np.array([])
        for stage, Q in gain_CIC_specs.items():
            CIC_stages = np.random.poisson(Q, size=array_size)
            # loop through, numel_gain_register entries at a time
            # Each of the numel_gain_register stage_chunks has length array_size
            stage_chunks.append(CIC_stages)
            stage_numbers = np.append(stage_numbers, stage)
        
    for i in range(len(stage_numbers)):
       gain = (1+P)**stage_numbers[i] 
       inds = np.where(stage_chunks[i] > 0)
       if inds[0].size > 0:
           n_out = rand_em_gain(stage_chunks[i][inds], gain, stage_numbers[i], threshold)
           partial_CIC[inds] = partial_CIC[inds] + n_out
    
    return partial_CIC


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    if False:
        partial_cic1 = partial_CIC(array_size=100, em_gain=1000, numel_gain_register=604, 
                                gain_CIC_Q=None, gain_CIC_specs={1:.001, 400:.09}, threshold=1e8)

        Q1 = (1000**(1/604)-1)/10
        partial_cic2 = partial_CIC(array_size=100, em_gain=1000, numel_gain_register=604, 
                                gain_CIC_Q=(.001+.09)/604, gain_CIC_specs=None, threshold=1e8)

        plt.figure()
        plt.hist(partial_cic1.ravel(), bins=100, histtype='stepfilled', alpha=0.7)
        plt.hist(partial_cic2.ravel(), bins=100, histtype='stepfilled', alpha=0.7)
        plt.title('Histogram of partial CIC')
        plt.xlabel('Partial CIC output')
        plt.ylabel('Counts')
        plt.yscale('log')
        plt.grid(True)
    
    #g=500; n=2; M=604
    #g=3; n=10; M=604
    g=150; n=10; M=604
    nmax=int(g*n*2)
    x_arr = np.arange(0, nmax)
    gamma_arr = np.exp(_LogGamma(n,g,x_arr))
    plt.plot(x_arr, gamma_arr)
    plt.plot(x_arr, Pn(n,g,x_arr))
    BSprob = exact_gain_PDF(gain=g, M=M, w=n, x=x_arr)
    plt.plot(x_arr, BSprob)

    # mean estimation
    BSprob2 = np.zeros_like(BSprob)
    for i in range(len(BSprob2)):
        BSprob2[i] = BSprob[i]*i
    BSmean = np.sum(BSprob2)
    print('BS mean: ' + str(BSmean))

    gamma2 = np.zeros_like(gamma_arr)
    for i in range(len(gamma2)):
        gamma2[i] = gamma_arr[i]*i
    gamma_mean = np.sum(gamma2)
    print('gamma mean: ' + str(gamma_mean)) 
    #compare the two means
    print('Percent difference of means:  ' + str((BSmean - gamma_mean)/gamma_mean))
    
    # standard deviation comparison
    #2nd moment
    BSprob3 = np.zeros_like(BSprob)
    for i in range(len(BSprob3)):
        BSprob3[i] = BSprob[i]*i**2
    BSstd = np.sqrt(np.sum(BSprob3) - BSmean**2)
    print('BS std dev: ' + str(BSstd)) 

    gamma3 = np.zeros_like(gamma_arr)
    for i in range(len(gamma3)):
        gamma3[i] = gamma_arr[i]*i**2
    gamma_std = np.sqrt(np.sum(gamma3) - gamma_mean**2)
    print('gamma std dev: ' + str(gamma_std)) 


    n=100;g=1000
    x_arr = np.round(np.linspace(n, n*g*4, num=4000)).astype(int)
    plt.plot(x_arr, np.exp(_LogGamma(n,g,x_arr)))
    plt.plot(x_arr, Pn(n,g,x_arr))

    #OLD COMMENT
    # Generally, the agreement b/w the old and new methods is good.  The new
    # method just speeds up the code a lot, especially when cosmics are present.
    # Old method functions below:

    def _apply_gain_old(n_in_array, em_gain, max_out):
        """Apply a specific em_gain to all nonzero n_in values."""
        # Initialize output count array
        n_out_array = np.zeros_like(n_in_array)

        # Get unique nonzero n_in values
        n_in_unique = np.unique(n_in_array)
        n_in_unique = n_in_unique[n_in_unique > 0]

        # Generate random numbers according to the gain distribution for each n_in
        for n_in in n_in_unique:
            inds = np.where(n_in_array == n_in)[0]
            n_out_array[inds] = _rand_pdf_old(n_in, em_gain, max_out, len(inds))

        # n_out_array = np.random.gamma(n_in_array, em_gain)
        # n_out_array = np.round(n_out_array)

        return n_out_array


    def _rand_pdf_old(n_in, em_gain, x_max, size):
        """Draw samples from the EM gain distribution."""
        x = np.random.random(size)

        # Use exact solutions for n_in == 1 and 2
        if n_in == 1:
            n_out = -em_gain * np.log(1 - x)
        elif n_in == 2:
            n_out = -em_gain * special.lambertw((x-1)/np.exp(1), -1).real - em_gain
        else:
            # For n > 2 use CDF approximation
            # Use x values ranging from 0 to maximum allowable x output
            x_axis = np.arange(0, x_max).astype(float)
            x_axis[0] = np.finfo(float).eps  # Use epsilon to avoid divide by 0
            cdf = _get_cdf(n_in, em_gain, x_axis)

            if cdf is None:
                # If cdf maxes out, return maximum value
                n_out = np.ones_like(x) * x_max
            else:
                # Draw random samples from the CDF
                cdf_lookups = (cdf.max() - cdf.min()) * x + cdf.min()
                n_out = x_axis[np.searchsorted(cdf, cdf_lookups)]  # XXX This could be made more accurate

        return np.round(n_out)


    def _get_cdf(n_in, em_gain, x):
        """Return an approximate CDF for the EM gain distribution.

        Basden 2003 probability distribution function is as follows:

            pdf = x^(n_in-1) * exp(-x/g) / (g^n_in * factorial(n_in-1))

        """
        # Because of the cancellation of very large numbers, first work in log space
        logpdf = (n_in-1)*np.log(x) - x/em_gain - n_in*np.log(em_gain) - special.gammaln(n_in)
        pdf = np.exp(logpdf)

        # XXX This is a rough but safe solution
        sum_pdf = np.sum(pdf)
        if sum_pdf == 0:
            cdf = None
        else:
            cdf = np.cumsum(pdf / sum_pdf)

        return cdf

    def compare_stats(g, n, n_samples, max_val, num_bins, plot=False, threshold=1e7):

        n_in_array = np.array([n]*n_samples)
        old_method = rand_em_gain(n_in_array, g, 604, threshold=0) 
        #old_method = _apply_gain_old(n_in_array, g, max_val)

        # new method:
        x = rand_em_gain(n_in_array, g, 604, threshold=threshold)

        print("For n={}, g={}:".format(n,g))
        print('Mean for old method:  ', np.mean(old_method))
        print('Std dev for old method:  ', np.std(old_method))
        print('Mean of new method:  ', np.mean(x))
        print('Std dev for new method:  ', np.std(x))
        print('Difference of means:  ', np.mean(old_method) - np.mean(x))
        print('Percentage of std dev for difference:  ', (np.mean(old_method) - np.mean(x))/(g*n))
        print('theoretical mean (gamma):  ', g*n)
        print('theortical std dev (gamma):  ', g*np.sqrt(n))
        print()

        if plot==True:
            fig, ax = plt.subplots()
            H = ax.hist(old_method, bins = num_bins)
            ax.set_ylabel('number of occurrences')
            ax.set_xlabel('gained counts (e-)')
            ax.set_title('Histogram of Gained Counts (Old Method, n={})'.format(n))

            fig, ax = plt.subplots()
            H = ax.hist(x, bins = num_bins)
            ax.set_ylabel('number of occurrences')
            ax.set_xlabel('gained counts (e-)')
            ax.set_title('Histogram of Gained Counts (New Method, n={})'.format(n))

            plt.show()

    g = 2 #20 #200
    n_samples = 10000
    #max_val = 200000
    num_bins = 40
    # let max_out be the mean + 4*std dev from gamma dist for the max value
    # from serial_counts (using ENF ~ sqrt(2), which is fine even for low
    # gain since we just want an upper limit)
    def max_val(g, n):
        return g*n + 4*g*np.sqrt(2*n)
    
    threshold = 1e9

    # in original code, max_val used max(n_in_array) where that array was for
    # all serial cells; so artifically inflate by multiplying by 100
    n = 1
    np.random.seed(123) # for reproducibility
    compare_stats(g, n, n_samples, 100*max_val(g,n), num_bins, threshold=threshold)

    n2 = 2
    np.random.seed(123) # for reproducibility
    compare_stats(g, n2, n_samples, 100*max_val(g,n), num_bins, threshold=threshold)

    # now a value of n for which these methods differed
    n3 = 3
    np.random.seed(123) # for reproducibility
    compare_stats(g, n3, n_samples, 100*max_val(g,n), num_bins, threshold=threshold)

    n4 = 40
    np.random.seed(123) # for reproducibility
    compare_stats(g, n4, n_samples, 100*max_val(g,n), num_bins, threshold=threshold)

    n5 = 100
    np.random.seed(123) # for reproducibility
    compare_stats(g, n5, n_samples, 100*max_val(g,n), num_bins, threshold=threshold)

    n6 = 100
    g6 = 100
    np.random.seed(123) # for reproducibility
    compare_stats(g6, n6, n_samples, 100*max_val(g6,n6), num_bins, threshold=threshold)

    n7 = 1000
    g7 = 1000
    np.random.seed(123) # for reproducibility
    compare_stats(g7, n7, n_samples, 100*max_val(g7,n7), num_bins, threshold=threshold)

    n8 = 18000
    g8 = 5000
    np.random.seed(123) # for reproducibility
    compare_stats(g8, n8, n_samples, 100*max_val(g8,n8), num_bins, threshold=threshold)

    #g=100;M=604;nmax=4000;w=3
    #prob = Brian_Sutin(gain=1000, M=604, nmax=4000, w=3)
    g=100;M=50;nmax=1000;w=3
    prob = exact_gain_PDF(gain=g, M=M, w=w, x=np.arange(0, nmax))
    plt.figure()
    plt.semilogy(np.arange(0, len(prob)), prob)
    plt.semilogy(np.arange(0, len(prob)), np.exp(_LogPn(w, g, np.arange(0, len(prob)))))
    plt.semilogy(np.arange(0, len(prob)), np.exp(_LogGamma(w,g,np.arange(0, len(prob)))))
    plt.title('log plot of Sutin for gain=' + str(g) + ', \n number of stages=' + str(M) + ', number of incoming=' + str(w))
    plt.xlabel('output value index')
    plt.ylabel('probability')
    plt.grid(True)
    plt.show()

    pass