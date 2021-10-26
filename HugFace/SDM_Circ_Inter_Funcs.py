import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix
import pandas as pd
from scipy.integrate import quad
import time
from scipy.special import comb

## Circle intersection approximation as given in the Appendix of the Sparse Distributed Memory book (Kanerva 1988). We approximate the integral using SciPy.

def softmax(x, beta):
    assert len(x.shape) <3, 'this softmax can currently only handle vectors'
    x = x * beta
    return np.exp(x)/np.exp(x).sum()

def f(x, c_p):
    return 1/(2*np.pi*np.sqrt(x*(1-x)))*np.exp(-0.5*(c_p**2/(1-x)))

def optimal_p_and_ham(m,r,n):
    optimal_p = 1/(2*m*r)**(1/3)
    d = space_frac_to_hamm_dist(n, [optimal_p])[0]
    return optimal_p, d

def fit_beta_regression(n, dvals, res):
    dvals = np.asarray(dvals)
    xvals = 1-(2*dvals)/n
    res = np.asarray(res)
    zeros_in_res = False
    if res[-1] == 0.0:
        print("res equals 0, problem for the log. removing from the equation here.")
        mask = res!=0.0
        num_zeros = (res==0.0).sum()
        res = res[mask]
        xvals = xvals[mask]
        zeros_in_res = True
    yvals = np.log(np.asarray(res))
    # log linear regression closed form solution. 
    beta = np.cov(xvals, yvals)[0][1] / np.var(xvals)
    b = np.mean(yvals) - beta*np.mean(xvals)
    fit_beta_res = softmax(xvals, beta)
    #print(fit_beta_res)
    #mse between res and beta res: 
    print('MSE:',np.sum((res-fit_beta_res)**2) )
    
    print('normalized MSE:',np.sum(((res/sum(res))-(fit_beta_res/sum(fit_beta_res)))**2) )
    
    if zeros_in_res: 
        # only true if there were 0 res values. need to append 0s to the end
        fit_beta_res = np.append(fit_beta_res, np.zeros(num_zeros))
    
    return fit_beta_res, beta


def expected_intersection_lune(n, dvals, hamm_dist, r):
    """
    Computes the fraction of the space that exists in the circle intersection using the Lune equation. 
    
    args::
    n = space dimension
    dvals = Hamm dist between circle centers
    hamm_dist = hamming distance radius each circle uses
    r = number of neurons
    
    hard_mem_places = turns the fraction of the space in the 
    expected number of neurons
    that exist in this fraction. 
    
    ------------
    
    returns:: 
    res = list of floats for fraction of the space
    """
    
    
    #ensure all are ints: 
    n = int(n)
    hamm_dist = int(hamm_dist)
    r = int(r)
    
    perc_addresses_w_neurons = r/(2**n) 
    
    res = []
    area = 0
    # compute size of circle
    for i in range(hamm_dist+1):
        area += comb(n,i)

    for d in dvals: 
        # compute lune
        d = int(d)
        lune = 0
        for i in range(d):
            j = i+1
            if j%2==0:
                continue
            lune+= comb(j-1, (j-1)/2)*comb(n-j, hamm_dist-((j-1)/2))
        intersect = area - lune
        #print(d, intersect, area, lune, perc_addresses_w_neurons)
        expected_intersect = np.log(intersect)+np.log(perc_addresses_w_neurons)
        res.append(np.exp(expected_intersect))
        
        '''if d==10:
            print(intersect)
            print(np.exp(expected_intersect))'''
    res = np.asarray(res)
    return res

def expected_intersection_continuous(n, dvals, hamm_dist, r, hard_mem_places):
    """
    Computes the fraction of the space that exists in the circle intersection using the continuous approximation to the Lune equation. 
    
    args::
    n = space dimension
    dvals = Hamm dist between circle centers
    hamm_dist = hamming distance radius each circle uses
    r = number of neurons
    
    hard_mem_places = turns the fraction of the space in the 
    expected number of neurons
    that exist in this fraction. 
    
    ------------
    
    returns:: 
    res = list of floats for fractions of the space or number of neurons present in this fraction depending if hard_mem_places is on.
    """

    res = []
    for dv in dvals:
        c_p = (hamm_dist-(n/2))/np.sqrt(n/4)
        intersect = quad(f, dv/n,1, args=(c_p))
        num = intersect[0]
        if hard_mem_places:
            num*=r
        res.append(num)
    return res
    
def space_frac_to_hamm_dist(n, space_frac_rang):
    """ Computes the Hamming distance that should be used for a circle 
    to have an area that includes a given fraction of a given n 
    dimensional space.
    
    args::
    - n = space dimension
    - space_frac_rang = list of space fractions to use
    
    returns::
    -list of hamming distances to use
    """
    
    hamm_distances = []
    for space_frac in space_frac_rang:
        hamm_distances.append( int(binom.ppf(space_frac, n, 0.5)) )
    return hamm_distances

def hamm_dist_to_space_frac(n, hamm_dist_rang):
    """ Computes the space fraction $p$ that corresponds to a given Hamming distance input
    
    args::
    - n = space dimension
    - space_frac_rang = list of Hamming distances used
    
    returns::
    - list of p fractions
    """
    
    pfracs = []
    for hd in hamm_dist_rang:
        pfracs.append( binom.cdf(hd, n, 0.5) )
    return pfracs
    