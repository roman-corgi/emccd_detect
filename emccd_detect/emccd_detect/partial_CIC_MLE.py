"""This script has all the functions needed to perform maximum likelihood 
analysis to determine the probability distribution that fits the data best.  
This script reproduces the results in the paper when the sample data from the 
paper is run. The sample data are dark frames with a 1-second exposure time 
at -85 degrees C from the T-e2v 301 EMCCD, and 
these average values for the whole data set were expected:  
EM gain 5000, k gain (i.e., e- per DN): 8, read noise: 110 e-

If you download the sample data in a folder called "data_dir", and if it is in 
the same location as this script, running this script as is will reproduce the 
paper's results.  The function calls at the bottom include a "cut" that ignores 
the many zero values of the histogram which do not contribute to the fit. 
The program will run fairly quickly over the MLE optimizations
that do not include partial CIC, but they may take a few hours for the 
optimizations that account for partial CIC.

https://arxiv.org/abs/2405.17622
 
https://doi.org/10.1117/1.JATIS.11.1.018005
Journal reference:	J. Astron. Telesc. Instrum. Syst. 11(1), 018005 (2025)

-by Kevin Ludwick
"""

import numpy as np
from scipy.special import (gamma, factorial,
                           hyp2f1, hyp0f1, gammaln)
from scipy.optimize import minimize, Bounds
from scipy.interpolate import UnivariateSpline
from astropy.io import fits
from scipy.stats import chisquare, chi2

def _LogPn(n, g, x):
    '''The log of the EM gain PDF, valid for small and large gain values.  
    Normalizing each term (n-dependent normalization) is important to avoid 
    negative values.'''
    x = x.astype(float)
    out = np.ones_like(x).astype(float)*(-np.inf) # log of 0: -inf
    good_ind = np.where(x >= n)
    X = x[good_ind]
    #Pn = ((2 - n + X)*gamma(n + X))/(np.e**(X/g)*g**n*gamma(n)*gamma(2 + X))
    # I reduced some gammas from top and bottom
    if n == 1:
        LogPn = np.log(2 - n + X) - (X/g + n*np.log(g) + np.log(X+1))
    elif n == 2:
        LogPn = np.log(2 - n + X) - (X/g + n*np.log(g))
        #Pn = ((2 - n + X))/(np.e**(X/g)*g**n)
    elif n == 3:
        LogPn = np.log(2 - n + X) + np.log(X+2) - (X/g + n*np.log(g))
        #Pn = ((2 - n + X)*(X+2))/(np.e**(X/g)*g**n)
    elif n > 3:
    # way to handle the large gammas that cancel on top and bottom without
    # getting a nan:  func = gamma(n+X)/gamma(2+X) simplified for Python
        #func = np.array([i+X-1 for i in range(3, n+1)]).T
        #LogPn = (np.sum(np.log(func),axis=1) + np.log(2-n+X) - (X/g + n*np.log(g)))
        logfunc = gammaln(n+X) - gammaln(2+X)
        # Pn = (np.product(func,axis=1)*((2 - n + X))/
        #       (np.e**(X/g)*g**n))
        LogPn = (logfunc + np.log(2-n+X) -
                (X/g + n*np.log(g)))

    # Pnnorm = ((4**n*gamma(0.5 + n)*(np.e**(1/g)*
    #         hyp2f1(1,2*n,2 + n,np.e**(-1/g))/((n+1)*n) +
    #         hyp2f1(2,1 + 2*n,3 + n,np.e**(-1/g))/((n+2)*(n+1))))/
    #         (np.e**((1 + n)/g)*g**n*np.sqrt(np.pi)))
    LogPnnorm = (n*np.log(4) + gammaln(0.5+n) - np.log(n+1) +
                 np.log(np.e**(1/g)*
            hyp2f1(1,2*n,2 + n,np.e**(-1/g))/(n) +
            hyp2f1(2,1 + 2*n,3 + n,np.e**(-1/g))/(n+2)) -
            ((1+n)/g + n*np.log(g) + 1/2*np.log(np.pi)))
    if np.isinf(LogPnnorm):
        #use Stirling's approx on gamma
        LogPnnorm = (n*np.log(4) + 1/2*(np.log(2*np.pi)-np.log(0.5+n)) +
                     (0.5+n)*np.log((0.5+n)/np.e) - np.log(n+1) +
                     np.log(np.e**(1/g)*
            hyp2f1(1,2*n,2 + n,np.e**(-1/g))/(n) +
            hyp2f1(2,1 + 2*n,3 + n,np.e**(-1/g))/(n+2)) -
            ((1+n)/g + n*np.log(g) + 1/2*np.log(np.pi)))
    out[good_ind] = LogPn - LogPnnorm 
    # nan can happen when Pnnorm gets negative (and it does for large n)
    outnan= np.where(np.isnan(out))
    Xnan = x[outnan]
    # in this case, usual gamma distribution is fine
    out[outnan] = _LogGamma(n, g, Xnan)

    return out


def _PoissonPn(L, g, x):
    '''The composition of the Poisson and the EM gain PDF.'''
    x = x.astype(float)
    out = np.zeros_like(x).astype(float)
    xless1 = np.where(x < 1)
    xless0 = np.where(x < 0)
    out[xless1] = np.exp(-L) #Poisson(L, x=0)

    xgreat1 = np.where(x >= 1)
    X = x[xgreat1]
    # do sum of terms a few sigma around L
    if L>1:
        R = np.round(np.arange(max(L-5*np.sqrt(L), 1), max(L+5*np.sqrt(L)+1, 2+1))).astype(int)
    else:
        R = np.arange(1, 5)
    if g > 1:
        prelim = np.array([np.exp(_LogPn(i,g,X) + _LogPoisson(L,np.array([i]))) for i in
                        R]).T
        out[xgreat1] = np.sum(prelim, axis=1)
        # for values that go to 0, use Gamma dist
        out0 = np.where(out == 0)
        out[out0] = np.exp(_LogPoissonGamma(L, g, x[out0]))
    else:
        out[xgreat1] = np.exp(_LogPoisson(L, X))
    # nans can occur very near g=1 or for very big x values, and they all
    # should be just 0
    outnan = np.where(np.isnan(out))
    out[outnan] = 0
    out[xless0] = 0 # overwrites any that were written above
    return out


def _W(p, M, g, x):
    '''The partial CIC PDF (for 0 incoming particles to the gain register.)
    Eq. 22 of the paper.'''
    x = x.astype(float)
    W =np.zeros_like(x).astype(float)
    xless1 = np.where(x<1)
    xgreat1 = np.where(x>=1)
    Xgreat1 = x[xgreat1]
    W[xless1] = (1-p)**M
    xless0 = np.where(x<0)
    W[xless0] = 0
    # r=1 case taken care of separately:
    logw_r1 = -(Xgreat1/g**(1/M)) + np.log(((1 - p)**(-1 + M)*p)/g**(1/M))
    
    pieces = M+5
    Marray = np.arange(2, M+1, pieces)
    if Marray[-1] != M+1:
        Marray = np.append(Marray, M+1)
    Marray = Marray.astype(int)
    wsum = np.zeros((Marray.size-1, Xgreat1.size))
    for m in range(len(Marray)-1):

        w = np.array([( np.exp(-(Xgreat1/g**((1 + r)/(2.*M))))*(((1 - p)**(-2 + M)
        *p*(g**((1 + r)/(2.*M))*(1 - p)*
            hyp2f1(1 - r,1 + Xgreat1,1,p/(g**((1 + r)/(2.*M))*(-1 + p))) +
            p*(1 - r)*hyp2f1(2 - r,2 + Xgreat1,2,p/
                                (g**((1 + r)/(2.*M))*(-1 + p)))))/
        g**((1 + r)/M))) for r in range(Marray[m], Marray[m+1])])
        wsum[m] = np.sum(w, axis=0)


    Wgreat1 = np.sum(wsum, axis=0)
    Wgreat1 = Wgreat1 + np.exp(logw_r1)
    W[xgreat1] = Wgreat1

    out = W 
    return out

def _Wmean(p, M, g, upper):
    '''The expected mean of the partial CIC PDF.'''
    x=np.arange(1,upper)
    Wmean = sum(x*_W(p,M,g,x)/_Wnorm(p,M,g))
    return Wmean

def _Wstd(p, M, g, upper):
    '''The standard deviation associated with the partial CIC PDF.'''
    x=np.arange(1,upper)
    Wpart = sum(x**2*_W(p,M,g,x)/_Wnorm(p,M,g))
    Wmean = _Wmean(p,M,g,upper)
    Wstd = np.sqrt(Wpart - Wmean**2)
    return Wstd

def _Wnorm(p, M, g):
    '''The normalization factor for the partial CIC PDF.'''
    _Wnorm = lambda y: _W(p, M, g, y)
    Wnorm = _infsum(_Wnorm, 1, 1e-16, 500)
    return Wnorm + (1-p)**M # x=0 case included



def _PoissonPartCICRNFFT_W(L,Q,g,M, x, l, rn, mu, xbounds=None, cut=None):
    '''The PDF represented by Eq. 31, accounting for partial CIC.  
    L is the mean of the Poisson PDF, Q is the probability of a CIC in a gain
    stage, g is the gain, M is the number of gain stages, x is the number of 
    electron counts, rn is the read noise, and mu is the mean of the normal
    distribution for read noise.
    Terms for n=1 through n=l calculated here.  
    l >=3 should be sufficient for high-gain frames.'''
    sp = 1
    if xbounds is None:
        xmin = x.min() - 1000
        xmax = x.max() + 1000
    else:
        xmin = xbounds[0]
        xmax = xbounds[1]
    xrange = np.arange(xmin, xmax, sp)

    PGpad = _PoissonPartCIC_W(L,Q,g,M,xrange,l)
    rnpad = np.exp(_log_rn_dist(rn,mu,xrange))
    PGfft = np.fft.fft((PGpad))
    rnfft = np.fft.fft((rnpad))
    fx = np.fft.fftfreq(PGpad.shape[0])
    convp = np.abs(np.fft.ifft(PGfft*rnfft*np.exp(2j*np.pi*fx*(-xrange.min()*1/sp))))
    conv = convp
    if conv.min() <= 0 or np.isnan(conv.any()):
        # read noise so distinct from other dist that it doesn't register;
        # so go back to just other dist
        conv = _PoissonPartCIC_W(L,Q,g,M,xrange,l)
    norm = None
    if cut is not None:
        ind = np.argmin(np.abs(xrange-cut))
        norm = np.sum(conv[ind:])
    interp = UnivariateSpline(xrange, conv, s = 0, ext=3)
    out = interp(x)
    #Sometimes the interpolated curve dips slightly below 0 if the curve is 
    # close enough to 0, which is not appropriate for a PDF.  Corrected here 
    # by setting to the first non-zero minimum.
    if out[out>0].size > 0:
        out[out<=0] = np.min(out[out>0])
    else:
        out[out<=0] = 0
    return out, norm 

def _PoissonPartCIC_W(L, Q, g, M, x, l):
    '''Composition of partial CIC + EM gain PDF with Poisson PDF (Poisson for
    electrons in the pixels, before getting to gain register).  Eq. 28 from 
    the paper.  L is the mean value for the Poisson distribution, 
    Q is the probability of a CIC in a gain stage, g is the gain, M is the number
    of gain stages, x is the electron count to evaluate the PDF at, 
    and l is the number of terms to use in the sum in Eq. 28.'''
    if Q <= 0 and g > 1:
        out = _PoissonPn(L, g, x)
    elif g <= 1: # physically, p > 0 only if gain > 1
        out = np.exp(_LogPoisson(L, x))
    else:
        x = x.astype(float)
        out = np.zeros_like(x).astype(float)

        Wnorm = _Wnorm(Q, M, g)
        xless1 = np.where(x < 1)
        out[xless1] = np.exp(-L)*(1-Q)**M/Wnorm

        xless0 = np.where(x < 0)
        out[xless0] = 0

        xgreat0 = np.where(x >= 0)
        X = x[xgreat0]
        if l < 1:
            raise ValueError('l must be an integer and >= 1.')
        SUM = np.zeros_like(X).astype(float)
        sp=1
        xrange = np.arange(0, X.max()+100, sp)
        PGpad = _W(Q,M,g,xrange)/Wnorm
        PGfft = np.fft.fft((PGpad))
        fx = np.fft.fftfreq(PGpad.shape[0])

        for i in range(1, l+1): # sum includes i=l
            rnpad = np.exp(_LogPn(i,g,xrange))
            rnfft = np.fft.fft((rnpad))
            conv = np.abs(np.fft.ifft(PGfft*rnfft*np.exp(2j*np.pi*fx*(-xrange.min()*1/sp))))
            interp = UnivariateSpline(xrange, conv, s = 0, ext=3)
            interpX = interp(X)
            if interpX[interpX>0].size > 0:
                interpX[interpX<=0] = np.min(interpX[interpX>0])
            else:
                interpX[interpX<=0] = 0
            SUM = SUM + interpX*np.exp(_LogPoisson(L,np.array([i])))
        out[xgreat0] = np.exp(-L)*_W(Q, M, g, X)/Wnorm + SUM

    return out



def _rn_dist(rn, mu, x):
    '''Normal distribution for read noise, centered at mu with standard
    deviation rn.'''
    x = x.astype(float)
    rn_dist = np.exp(-(x-mu)**2/(2*rn**2))/(np.sqrt(2*np.pi)*rn)
    return rn_dist


def _log_rn_dist(rn, mu, x):
    '''Log of Normal distribution for read noise, centered at mu with standard
    deviation rn.'''
    x = x.astype(float)
    log_rn_dist = -(x-mu)**2/(2*rn**2) - np.log(np.sqrt(2*np.pi)*rn)
    return log_rn_dist



def EM_gain_fit_W(frames, Nem, l, fluxe, pCIC, gain,
                read_noise, rn_mean,
                gmax, tol=1e-10, lthresh=0, cut=None):
    '''Performs maximum likelihood estimation (MLE).  Finds the parameter 
    values which maximize the likelihood of the overall PDF (including partial 
    CIC) of what is read 
    out from a EMCCD (Eq. 31 of the paper.)  The variables represented by 
    fluxe, pCIC, gain, read_noise, and rn_mean are allowed to vary.
    
    Parameters
    ----------
    frames : array-like
        Array containing data from a frame or frames with non-unity EM gain.

    Nem : int
        Number of gain registers. 

    l : int
        Number of terms in the sum from Eq. 28 to use.
        
    fluxe : float
        Mean number of electrons per pixel expected to be present in frames.
        This parameter is used as the initial guess for the mean for the
        Poisson distribution for the optimization process.  >= 0.
    
    pCIC : float
        Initial guess for the optimization process for the probability
        of a clock-induced charge (CIC) produced in a 
        gain stage. The paper calls this Q.

    gain : float
        Initial guess for the optimization process for the EM gain applied for
        frames.  >= 1.

    read_noise : float
        Initial guess for the optimization process for the read noise in frames.

    rn_mean : float
        Initial guess for the optimization process for the mean of the normal 
        distribution for the read noise. 

    gmax : float
        Upper bound of EM gain for the MLE fit value for gain.

    tol : float, optional
        Tolerance used in the MLE analysis, used in scipy.optimize.minimize.
        Defaults to 1e-16.

    lthresh : float, optional
        The minimium frequency for a histogram bin for the data from
        frames that is used for MLE analysis.  >= 0.    

    cut : float, optional
        If the user wants to apply MLE over a subset of the domain of the
        probability distribution, the user can specify cut, which takes the
        domain above this value and normalizes the probability over that
        region.  This is useful if part of the histogram is not useable below a
        certain count level.  Defaults to None, which means no cut is employed.
    '''
    f = np.ravel(frames)
    y_vals, bin_edges = np.histogram(f, bins=int(np.round(f.max()-f.min())))
    x_vals = bin_edges[:-1]
    good_ind = np.where(y_vals > lthresh)
    yv = y_vals[good_ind]
    xv = x_vals[good_ind]

    if cut is not None:
        cut = min(cut, xv.max())
        gind = np.where(xv >= cut)
        yv = yv[gind]
        xv = xv[gind]

    P = gain**(1/Nem) - 1
    bounds = Bounds(lb=np.array([f.mean()/(2*gain), 0, max(1+np.finfo(float).eps, gain*.5), read_noise*0.5, -read_noise*.5]),
                ub=np.array([max(2*f.mean()/gain, 1), P, min(gmax, gain*1.5), read_noise*1.5, read_noise*.5]))

    def _loglik(v):
        '''The negative of the log of the likelihood.'''
        L, p, g, rn, mu = v
        ar, norm = _PoissonPartCICRNFFT_W(L,p,g,Nem,xv,l,rn,mu, xbounds=(x_vals.min(), x_vals.max()), cut=cut)
        logar = np.log(ar)
        if cut is not None:
            logar = logar - np.log(norm)
        lik_ar = yv*logar 
        output = -np.sum(lik_ar)
        print('L,p,g,rn,mu:  ', L,p,g,rn,mu)
        print('lik:  ', output)

        return output

    res = minimize(fun=_loglik,
            x0=np.array([fluxe, pCIC, gain, read_noise, rn_mean]),
            bounds=bounds,
            tol=tol,
            )
    out, _ = _PoissonPartCICRNFFT_W(res.x[0],res.x[1],res.x[2],Nem,xv,l,res.x[3],res.x[4],xbounds=(x_vals.min(), x_vals.max()))
    scale = np.sum(yv)/np.sum(out)
    chisquare_value, pvalue = chisquare(yv/scale, out)
    print('Maximum log-likelihood: ', -res.fun)
    print('chi square value:  ', chisquare_value, ', p value: ', pvalue)
    print('critical chi-square value:  ', chi2.ppf(1-0.05, df=xv.max()))

    return res, chisquare_value, pvalue


def EM_gain_fit_LPG_W(frames, Nem, l, fluxe, pCIC, gain,
                read_noise, rn_mean,
                gmax, tol=1e-10, lthresh=0, cut=None):
    '''Performs maximum likelihood estimation (MLE).  Finds the parameter 
    values which maximize the likelihood of the overall PDF (including partial 
    CIC) of what is read 
    out from a EMCCD (Eq. 31 of the paper.)  The variables represented by 
    fluxe, pCIC, gain, read_noise, and rn_mean are allowed to vary.
    
    Parameters
    ----------
    frames : array-like
        Array containing data from a frame or frames with non-unity EM gain.

    Nem : int
        Number of gain registers. 

    l : int
        Number of terms in the sum from Eq. 28 to use.
        
    fluxe : float
        Mean number of electrons per pixel expected to be present in frames.
        This parameter is used as the initial guess for the mean for the
        Poisson distribution for the optimization process.  >= 0.
    
    pCIC : float
        Initial guess for the optimization process for the probability
        of a clock-induced charge (CIC) produced in a 
        gain stage. The paper calls this Q.

    gain : float
        Initial guess for the optimization process for the EM gain applied for
        frames.  >= 1.

    read_noise : float
        The fixed value used for the read noise in frames.  Parameter does not
        vary.

    rn_mean : float
        The fixed value used for the mean of the normal 
        distribution for the read noise.  Parameter does not vary.

    gmax : float
        Upper bound of EM gain for the MLE fit value for gain.

    tol : float, optional
        Tolerance used in the MLE analysis, used in scipy.optimize.minimize.
        Defaults to 1e-16.

    lthresh : float, optional
        The minimium frequency for a histogram bin for the data from
        frames that is used for MLE analysis.  >= 0.    

    cut : float, optional
        If the user wants to apply MLE over a subset of the domain of the
        probability distribution, the user can specify cut, which takes the
        domain above this value and normalizes the probability over that
        region.  This is useful if part of the histogram is not useable below a
        certain count level.  Defaults to None, which means no cut is employed.
    '''
    f = np.ravel(frames)
    y_vals, bin_edges = np.histogram(f, bins=int(np.round(f.max()-f.min())))
    x_vals = bin_edges[:-1]
    good_ind = np.where(y_vals > lthresh)
    yv = y_vals[good_ind]
    xv = x_vals[good_ind]

    if cut is not None:
        cut = min(cut, xv.max())
        gind = np.where(xv >= cut)
        yv = yv[gind]
        xv = xv[gind]

    P = gain**(1/Nem) - 1
    bounds = Bounds(lb=np.array([f.mean()/(2*gain), 0, max(1+np.finfo(float).eps, gain*.5)]),
                ub=np.array([max(2*f.mean()/gain, 1), P, min(gmax, gain*1.5)]))

    def _loglik(v):
        '''The negative of the log of the likelihood.'''
        L, p, g = v
        ar, norm = _PoissonPartCICRNFFT_W(L,p,g,Nem,xv,l,read_noise,rn_mean,
                                    xbounds=(x_vals.min(), x_vals.max()), cut=cut)
        logar = np.log(ar)
        if cut is not None:
            logar = logar - np.log(norm)
        lik_ar = yv*logar 
        output = -np.sum(lik_ar)
        print('L,p,g:  ', L,p,g)
        print('lik:  ', output)
        
        return output

    res = minimize(fun=_loglik,
            x0=np.array([fluxe, pCIC, gain]),
            bounds=bounds,
            tol=tol,
            )

    out, _ = _PoissonPartCICRNFFT_W(res.x[0],res.x[1],res.x[2],Nem,xv,l,read_noise,rn_mean,xbounds=(x_vals.min(), x_vals.max()))
    scale = np.sum(yv)/np.sum(out)
    chisquare_value, pvalue = chisquare(yv/scale, out)
    print('Maximum log-likelihood: ', -res.fun)
    print('chi square value:  ', chisquare_value, ', p value: ', pvalue)
    print('critical chi-square value:  ', chi2.ppf(1-0.05, df=xv.max()))

    return res, chisquare_value, pvalue


def _LogPoisson(L, x):
    '''Log of the Poisson distribution, with mean L.'''
    x = x.astype(float)
    out = np.zeros_like(x).astype(float)
    xless0 = np.where(x < 0)
    out[xless0] = -np.inf
    xgreat0 = np.where(x>=0)
    X = x[xgreat0]
    #out[xgreat0] = -L + X*np.log(L) - np.log(factorial(X))
    # for large X, log of factorial will be inf, so use gammaln, which is the log of the gamma function, and for integer X, gamma(X) = factorial(X-1)
    out[xgreat0] = -L + X*np.log(L) - gammaln(X+1)
    # In case nans occur for huge x values, use approximation
    outinf = np.where(np.isinf(out))
    Xinf = x[outinf]
    out[outinf] = -L + Xinf*(1 + np.log(L) - np.log(Xinf)) - 0.5*np.log(2*np.pi*Xinf)

    return out


def _LogGamma(n, g, x):
    '''The log of the Gamma distribution for integer n (Erlang distribution) 
    for the gain PDF, where n is the number of incoming electrons to the 
    gain register and g is the gain. This is the log of Eq. 11.'''
    # n>=1 and integer; g>=1
    x = x.astype(float)
    out = np.zeros_like(x).astype(float)

    xgreat1 = np.where(x >= n)
    X = x[xgreat1]
    if g > 1:
        #out[xgreat1] = -(X/g) - n*np.log(g) + (-1 + n)*np.log(X) - np.log(factorial(-1 + n))
        # for large n, the log of factorial will be np.inf
        #better to use gammaln, which is the log of the gamma function, and for integer n, gamma(n) = factorial(n-1)
        out[xgreat1] = -(X/g) - n*np.log(g) + (-1 + n)*np.log(X) - gammaln(n)
        outinf = np.where(np.isinf(out))
        Xinf = x[outinf]
        # Stirling's approximation for large n
        out[outinf] = -(Xinf/g) - n*np.log(g) + (-1 + n)*np.log(Xinf) - \
                    (n*(-1 + np.log(n)) - np.log(n)/2 + np.log(2*np.pi)/2)

    xlessn = np.where(x < n)
    # there is analytic normalization we could apply, but it doesn't affect
    # the MLE process: as long as it's all 1 or less, and it is
    out[xlessn] = -np.inf # log(0)

    return out

def _LogPoissonGamma(L, g, x): 
    '''The log of the composition of the Poisson distribution with the 
    Erlang distribution, where L is the mean of the Poisson distribution and 
    g is the gain.  This is the log of Eq. 29.'''
    x = x.astype(float)
    out = np.zeros_like(x).astype(float)
    xless1 = np.where(x < 1)
    xless0 = np.where(x < 0)
    out[xless1] = -L

    xgreat1 = np.where(x >= 1)
    X = x[xgreat1]
    if g > 1:
        out[xgreat1] = -(L + X/g) + np.log(L/g) + np.log(hyp0f1(2,(L*X)/g))

        # In case nans occur for huge x values, use approximation
        outinf = np.where(np.isinf(out))
        Xinf = x[outinf]

        # do sum of terms a few sigma around L
        if L>1:
            R = np.arange(max(L-5*np.sqrt(L), 1), max(L+5*np.sqrt(L)+1, 2+1))
        else:
            R = np.arange(1, 5)
        logarray = np.array([(_LogGamma(i,g,Xinf)+_LogPoisson(L,np.array([i]))) for i in R]).T
        nn = np.exp(logarray)
        SUM = np.sum(nn, axis=1)
        out[outinf] = np.log(SUM)

        # if the above didn't work, then do rough estimate:
        # take the biggest log to represent the whole sum of
        # logs; exp it, but we want the log for the return anyways
        # So it's just the max log term (for the peak of the distribution)
        # times 2*std dev (for the rough full width), then take log of that:
        outinf2 = np.where(np.isinf(out))

        Xinf2 = x[outinf2]
        logarray2 = np.array([(_LogGamma(i,g,Xinf2)+_LogPoisson(L,np.array([i]))) for i in R]).T
        out[outinf2] = np.max(logarray2, axis=1) + np.log(2) + 0.5*np.log(L)


        out[xless0] = -np.inf # overwrites any that were written in line above
    else:
        out[xgreat1] = _LogPoisson(L, X)

    return out 


def _infsum(f, T, tol=1e-6, max_count=100):
    '''Performs an approximation of a infinite sum specified by the function of
    one variable f.  T is the spacing between terms chosen for summing, tol 
    is the amount the last term added on must be below in order for the 
    function to end its summing, and max_count is the number of terms at which
    to stop summing.  The function will stop summing when either the tolerance 
    tol is reached or when max_count is reached.'''
    if T <= 0:
        raise Exception("T must be bigger than 0.")
    counter = 0
    n, res = 0, 0
    while True:
        k = np.arange(T*(1+n), T*(2+n))
        term = np.sum(f(k))
        if (res+term)-res < tol or counter >= max_count:
            break
        n,res = n+1, res+term
        counter += 1
    return res


def _PoissonGammaConvFFT(L,g,x,rn, mu, xbounds=None):
    '''The PDF represented by Eq. 31, leaving out partial CIC (using Eq. 29).  
    L is the mean of the Poisson PDF, Q is the probability of a CIC in a gain
    stage, g is the gain, M is the number of gain stages, x is the number of 
    electron counts, rn is the read noise, and mu is the mean of the normal
    distribution for read noise.
    Terms for n=1 through n=l calculated here.  
    l >=3 should be sufficient for high-gain frames.'''
    sp=1
    if xbounds is None:
        xmin = x.min() - 1000
        xmax = x.max() + 1000
    else:
        xmin = xbounds[0]
        xmax = xbounds[1]
    xrange = np.arange(xmin, xmax, sp) 

    PGpad = np.exp(_LogPoissonGamma(L,g,xrange))
    rnpad = np.exp(_log_rn_dist(rn,mu,xrange))
    PGfft = np.fft.fft((PGpad))
    rnfft = np.fft.fft((rnpad))
    fx = np.fft.fftfreq(PGpad.shape[0])
    # include a shift to shift 0 to the min index position
    conv = np.abs((np.fft.ifft(PGfft*rnfft*np.exp(2j*np.pi*fx*(-xrange.min()*1/sp)))))

    if conv.min() <= 0 or np.isnan(conv.any()):
        # read noise so distinct from Poisson-gamma that it doesn't register;
        # so go back to just Poisson-gamma
        conv = np.exp(_LogPoissonGamma(L,g,xrange))
    interp = UnivariateSpline(xrange, conv, s = 0, ext=3)
    out = interp(x)
    #Sometimes the interpolated curve dips slightly below 0 if the curve is 
    # close enough to 0, which is not appropriate for a PDF.  Corrected here 
    # by setting to the first non-zero minimum.
    if out[out>0].size > 0:
        out[out<=0] = np.min(out[out>0])
    else:
        out[out<=0] = 0
    return out


def EM_gain_fit_conv(frames, fluxe, gain, gmax, rn, mu, divisor=1, tol=1e-10, lthresh=0, cut=None):
    '''Performs maximum likelihood estimation (MLE).  Finds the parameter 
    values which maximize the likelihood of the overall PDF represented by 
    Eq. 31, leaving out partial CIC (using Eq. 29).  
    The variables represented by fluxe and gain are allowed to vary.
    
    Parameters
    ----------
    frames : array-like
        Array containing data from a frame or frames with non-unity EM gain.

    fluxe : float
        Mean number of electrons per pixel expected to be present in frames.
        This parameter is used as the initial guess for the mean for the
        Poisson distribution for the optimization process.  >= 0.

    gain : float
        Initial guess for the optimization process for the EM gain applied for
        frames.  >= 1.

    gmax : float
        Upper bound of EM gain for the MLE fit value for gain.

    rn : float
        The fixed value used for the read noise in frames.  Parameter does not
        vary.

    mu : float
        The fixed value used for the mean of the normal 
        distribution for the read noise.  Parameter does not vary.
        
    divisor : float, optional
        The size of the range of integer values found in frames is divided by
        this parameter, and the result is used as the number of bins in the
        histogram.  Defaults to 1.

    tol : float, optional
        Tolerance used in the MLE analysis, used in scipy.optimize.minimize.
        Defaults to 1e-16.

    lthresh : float, optional
        The minimium frequency for a histogram bin for the data from
        frames that is used for MLE analysis.  >= 0.    

    cut : float, optional
        If the user wants to apply MLE over a subset of the domain of the
        probability distribution, the user can specify cut, which takes the
        domain above this value and normalizes the probability over that
        region.  This is useful if part of the histogram is not useable below a
        certain count level.  Defaults to None, which means no cut is employed.
    '''
    f = np.ravel(frames)
    y_vals, bin_edges = np.histogram(f, bins=int(np.round((f.max()-f.min())/divisor)))
    if divisor > 1:
        x_vals = (bin_edges + np.roll(bin_edges, -1))/2
    x_vals = bin_edges[:-1]
    good_ind = np.where(y_vals > lthresh)
    yv = y_vals[good_ind]
    xv = x_vals[good_ind]
    if cut is not None:
        gind = np.where(xv > cut)
        yv = yv[gind]
        xv = xv[gind]

    bounds = Bounds(lb=np.array([f.mean()/(2*gain), max(1+np.finfo(float).eps, gain*.5)]),
                ub=np.array([max(2*f.mean()/gain, 1), min(gain*1.5, gmax)]))

    def _loglik(v):
        '''The negative of the log of the likelihood.'''
        L, g = v
        ar = _PoissonGammaConvFFT(L,g,xv,rn,mu, xbounds=(x_vals.min(), x_vals.max()))
        logar = np.log(ar)
        if cut is not None:
            norm = np.sum(_PoissonGammaConvFFT(L,g,np.arange(cut,
                x_vals.max()),rn,mu,xbounds=(x_vals.min(), x_vals.max())))
            logar = logar - np.log(norm)
        out = -np.sum(yv*logar) 
        print('L,g:  ', L,g)
        print('lik: ', out)
        
        return out


    res = minimize(fun=_loglik,
            x0=np.array([fluxe, gain]),
            bounds=bounds,
            tol=tol,
            )
    out = _PoissonGammaConvFFT(res.x[0],res.x[1],xv,rn,mu,xbounds=(x_vals.min(), x_vals.max()))
    scale = np.sum(yv)/np.sum(out)
    chisquare_value, pvalue = chisquare(yv/scale, out)
    print('Maximum log-likelihood: ', -res.fun)
    print('chi square value:  ', chisquare_value, ', p value: ', pvalue)
    print('critical chi-square value:  ', chi2.ppf(1-0.05, df=xv.max()))

    return res, chisquare_value, pvalue


def EM_gain_fit_conv_rn(frames, fluxe, gain, gmax, rn, mu, divisor=1, tol=1e-10, lthresh=0, cut=None):
    '''Performs maximum likelihood estimation (MLE).  Finds the parameter 
    values which maximize the likelihood of the overall PDF represented by 
    Eq. 31, leaving out partial CIC (using Eq. 29).  
    The variables represented by fluxe, gain, rn, and mu are allowed to vary.
    
    Parameters
    ----------
    frames : array-like
        Array containing data from a frame or frames with non-unity EM gain.

    fluxe : float
        Mean number of electrons per pixel expected to be present in frames.
        This parameter is used as the initial guess for the mean for the
        Poisson distribution for the optimization process.  >= 0.

    gain : float
        Initial guess for the optimization process for the EM gain applied for
        frames.  >= 1.

    gmax : float
        Upper bound of EM gain for the MLE fit value for gain.

    rn : float
        The fixed value used for the read noise in frames.  Parameter does not
        vary.

    mu : float
        The fixed value used for the mean of the normal 
        distribution for the read noise.  Parameter does not vary.
        
    divisor : float, optional
        The size of the range of integer values found in frames is divided by
        this parameter, and the result is used as the number of bins in the
        histogram.  Defaults to 1.

    tol : float, optional
        Tolerance used in the MLE analysis, used in scipy.optimize.minimize.
        Defaults to 1e-16.

    lthresh : float, optional
        The minimium frequency for a histogram bin for the data from
        frames that is used for MLE analysis.  >= 0.    

    cut : float, optional
        If the user wants to apply MLE over a subset of the domain of the
        probability distribution, the user can specify cut, which takes the
        domain above this value and normalizes the probability over that
        region.  This is useful if part of the histogram is not useable below a
        certain count level.  Defaults to None, which means no cut is employed.
    '''
    f = np.ravel(frames)
    y_vals, bin_edges = np.histogram(f, bins=int(np.round((f.max()-f.min())/divisor)))
    if divisor > 1:
        x_vals = (bin_edges + np.roll(bin_edges, -1))/2
    x_vals = bin_edges[:-1]
    good_ind = np.where(y_vals > lthresh)
    yv = y_vals[good_ind]
    xv = x_vals[good_ind]
    if cut is not None:
        gind = np.where(xv > cut)
        yv = yv[gind]
        xv = xv[gind]

    bounds = Bounds(lb=np.array([f.mean()/(2*gain), max(1+np.finfo(float).eps, gain*.5), rn*0.5, -rn*.5]),
                ub=np.array([max(2*f.mean()/gain, 1), min(gmax, gain*1.5), rn*1.5, rn*.5]))

    def _loglik(v):
        '''The negative of the log of the likelihood.'''
        L, g, r, m = v
        ar = _PoissonGammaConvFFT(L,g,xv,r,m, xbounds=(x_vals.min(), x_vals.max()))
        logar = np.log(ar)
        if cut is not None:
            norm = np.sum(_PoissonGammaConvFFT(L,g,np.arange(cut,
                x_vals.max()),r,m,xbounds=(x_vals.min(), x_vals.max())))
            logar = logar - np.log(norm)
        lik_ar = yv*logar 
        out = -np.sum(lik_ar)
        print('lik: ', out)
        print('L, g, r, m:  ', L, g, r, m)
        
        return out

    res = minimize(fun=_loglik,
            x0=np.array([fluxe, gain, rn, mu]),
            bounds=bounds,
            tol=tol,
            )
    out = _PoissonGammaConvFFT(res.x[0],res.x[1],xv,res.x[2],res.x[3],xbounds=(x_vals.min(), x_vals.max()))
    scale = np.sum(yv)/np.sum(out)
    chisquare_value, pvalue = chisquare(yv/scale, out)
    print('Maximum log-likelihood: ', -res.fun)
    print('chi square value:  ', chisquare_value, ', p value: ', pvalue)
    print('critical chi-square value:  ', chi2.ppf(1-0.05, df=xv.max()))

    return res, chisquare_value, pvalue


if __name__ == '__main__':
    
    import os
    from pathlib import Path

    from cal.util.gsw_process import Process
    from cal.util.read_metadata import Metadata as MetadataWrapper
    import cal

    libfile = cal.__path__[0]
    here = os.path.abspath(os.path.dirname(__file__))
    meta_path = Path(libfile, 'util', 'metadata.yaml')
    # no actual nonlinearity correction assumed here
    nonlin_path = Path(libfile, 'util', 'testdata', 'ut_nonlin_array_ones.txt')
    meta = MetadataWrapper(meta_path)
    image_rows, image_cols, r0c0 = meta._unpack_geom('image')

    # full-well capacities
    fwc_em_e = 90000 #e-, for EM gain register
    fwc_pp_e = 50000 #e-, per pixel (before EM gain register)
    frametime = 1

    def read_in_files(directory, eperdn, bias_offset, gain, prescan=False):
        '''This function mainly subtracts the bias, divides by gain, and 
        converts from DN to e-.  See Process class methods for details 
        on the processing done for these specific EMCCD frames that are used
        for demonstration in this script. 
        '''
        proc = Process(bad_pix=np.zeros((meta.frame_rows,meta.frame_cols)), eperdn=eperdn,
                                fwc_em_e=fwc_em_e, fwc_pp_e=fwc_pp_e,
                                bias_offset=bias_offset, em_gain=gain,
                                exptime=frametime, nonlin_path=nonlin_path,
                                meta_path=meta_path)
        framelist = []
        for file in os.listdir(directory):
            if file.endswith('fits'):
                f = os.path.join(directory, file)
                d = fits.getdata(f)
                _, _, _, _, f0, b0, _ = proc.L1_to_L2a(d)
                # could just skip L2a_to_L2b(), but I like having a hook for
                # combining b0 with a proc_dark.bad_pix that isn't all zeros
                f1, b1, _ = proc.L2a_to_L2b(f0, b0)
                # to undo the division by gain in L2a_to_L2b()
                ff = f1*gain
                # assign NaN values to any pixels marked as bad (i.e., due to cosmic rays)
                ff = np.ma.masked_array(ff, mask=b1.astype(bool))
                f1 = ff.astype(float).filled(np.nan)
                # assign NaN values to last row, which does not have any physical readout
                f1[-1] = np.nan

                if prescan:
                    f1 = proc.meta.slice_section(f1, 'prescan')
                else:
                    f1 = proc.meta.slice_section(f1, 'image')
                framelist.append(f1)
        st = np.stack(framelist)
        return st

    
    # gain: 5000, eperdn: 8, read noise 110, darks
    directory = Path(here, 'data_dir')
    # gain input below shouldn't really matter since it is divided and then multiplied back in
    frames = read_in_files(directory, eperdn=8, bias_offset=0, gain=5000)

    # NOTE To run over the full frame used in the paper, use the line below:
    #frames = frames[1]
    # NOTE TO run over the 1 row used in the paper, uncomment the line below:
    frames = frames[0,500,:]

    s_out_ind = np.where(~np.isnan(frames))
    frames = frames[s_out_ind]

    # plots in the paper:
    if True: # change to True if plots desired
        import matplotlib.pyplot as plt
        x = np.arange(-750,1250)
        out, norm = _PoissonPartCICRNFFT_W(.01,0,5000,602, x, 2, 110, 0, xbounds=None, cut=None)
        out2, norm2 = _PoissonPartCICRNFFT_W(.01,0.001,5000,602, x, 2, 110, 0, xbounds=None, cut=None)
        out3, norm3 = _PoissonPartCICRNFFT_W(.01,0.003,5000,602, x, 2, 110, 0, xbounds=None, cut=None)
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(x, out, label='no partial CIC', linestyle='-')
        ax1.plot(x, out2, label='partial CIC (Q=0.001)', linestyle='--')
        ax1.plot(x, out3, label='partial CIC (Q=0.003)', linestyle='-.')
        ax1.set_title(r'$P_{rPg}$ Probability Distribution, Linear Scale')
        ax1.set_xlabel('Electron counts')
        ax1.set_ylabel('Count frequency')
        ax1.set_xlim(-500, 1250)
        ax1.legend()
        
        # Create inset of the log plot
        ax_inset = fig.add_axes([0.55, 0.35, 0.35, 0.35])
        ax_inset.semilogy(x, out, label='no partial CIC', linestyle='-')
        ax_inset.semilogy(x, out2, label='partial CIC (Q=0.001)', linestyle='--')
        ax_inset.semilogy(x, out3, label='partial CIC (Q=0.003)', linestyle='-.')
        ax_inset.set_title(r'Logarithmic Scale')
        ax_inset.set_xlabel('Electron counts')
        ax_inset.set_ylabel('Count frequency')
        ax_inset.legend()
        
        plt.show()

        if frames.size < 1E6:
            title = '1 row, commanded gain of 5000'
        else:
            title = '1 frame, commanded gain of 5000'
        y_vals, bin_edges = np.histogram(frames, bins=int(np.round((frames.max()-frames.min()))))
        x_vals = bin_edges[:-1]
        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals)
        ax.set_xlim(-1000, 2500)
        ax.set_xlabel('Electron counts')
        ax.set_ylabel('Count frequency')
        ax.set_title(title)

        # Create inset of the log plot
        ax_inset = fig.add_axes([0.55, 0.35, 0.35, 0.35])
        ax_inset.semilogy(x_vals, y_vals)
        ax_inset.set_xlabel('Electron counts')
        ax_inset.set_ylabel('Count frequency')
        ax_inset.set_title('Logarithmic Scale')

        plt.show()

    # No partial CIC, lambda and g (Fits 1 and 5 from paper)
    res1, chisquare_value1, pvalue1  = EM_gain_fit_conv(frames,.008, 5000,6000,110,0,cut=-800, tol=1e-10)
    # shown below:
    # res.success, res.x, res.fun, chisquare_value, pvalue 
    # frame 1 only:
    # (True, array([2.76514043e-02, 4.97152306e+03]), 6676631.432429293, 39.129370193111484, 1.0)
    # row index 500 only:
    # (True, array([1.67833954e-02, 4.99999998e+03]), 6444.748315364548, 0.7054126328932218, 1.0)

    # No partial CIC, lambda, g, sigma_rn, and mu (Fits 2 and 6 from paper)
    res2, chisquare_value2, pvalue2 = EM_gain_fit_conv_rn(frames,.008,5000,6000,110,0,lthresh=0,cut=-800, tol=1e-10)
    # shown below:
    # res.success, res.x, res.fun, chisquare_value, pvalue 
    # frame 1 only:
    # (True, array([1.90885768e-02, 4.86106525e+03, 1.15316038e+02, 5.31535511e+00]), 6666469.708461857, 6.25709059243572, 1.0)
    # row index 500 only:
    # (True, array([ 1.65201362e-02,  4.99996516e+03,  1.16149480e+02, -1.55274235e+01]), 6431.654075122935, 0.47416855117423884, 1.0)

    # With partial CIC, lambda, Q, and g (Fits 3 and 7 from paper)
    res3, chisquare_value3, pvalue3 = EM_gain_fit_LPG_W(frames, 604, 7, .005, .0005, 5000, 110, 0, 6000, cut=-800, tol=1e-10)
    # shown below:
    # res.success, res.x, res.fun, chisquare_value, pvalue 
    # frame 1 only:
    # (True, array([2.41292160e-02, 1.05875819e-03, 5.01924628e+03]), 6662394.908985481, 109.48038278397956, 1.0)
    # row index 500 only:
    # (True, array([1.67817595e-02, 0.00000000e+00, 4.99999999e+03]), 6444.7483025533, 0.7054144891891521, 1.0)

    # With partial CIC, lambda, Q, g, sigma_rn, and mu (Fits 4 and 8 from paper)
    res4, chisquare_value4, pvalue4 = EM_gain_fit_W(frames, 604, 7, .005, .0005, 5000, 110, 0, 6000, cut=-800, tol=1e-10)
    # shown below:
    # res.success, res.x, res.fun, chisquare_value, pvalue 
    # frame 1 only:
    # (True, array([1.23572894e-02, 1.05287008e-03, 5.05488780e+03, 1.13018960e+02, 3.01879091e+00]), 6659451.770300338, 34.95153139270946, 1.0)
    # row index 500 only:
    # (True, array([ 8.38176137e-03,  2.47298880e-03,  4.99993905e+03,  9.68414990e+01, -5.50000000e+01]), 6406.863944662827, 1.7811191256479115, 1.0)
