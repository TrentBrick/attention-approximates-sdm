"""
Author: Trenton Bricken @trentbrick

All functions in this script are used to generate and approximate the circle intersection in binary and continuous space. 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix
import pandas as pd
import scipy
from scipy.integrate import quad
import time
from scipy.special import comb
import torch
import torch.optim as optim
import torch.nn.functional as F

def softmax(x, beta):
    assert len(x.shape) <3, 'this softmax can currently only handle vectors'
    x = x * beta
    return np.exp(x)/np.exp(x).sum()

def cosine_to_hamm(cosines, n):
    return n*(1-cosines)/2

def hamm_to_cosine(hamms, n):
    return 1-(hamms*2/n)

def optimal_p_and_ham(m,r,n): # for SNR
    """ Gives the SNR optimal hamming distance and corresponding space fraction."""
    optimal_p = 1/(2*m*r)**(1/3)
    d = space_frac_to_hamm_dist(n, [optimal_p])[0]
    return optimal_p, d

### FUNCTIONS APPROXIMATING A KNOWN AND PROVIDED CIRCLE INTERSECTION: 

def fit_beta_regression(n, xvals, res, return_bias=False, ham_input=True):
    """ Log linear regression to fit a beta coefficent to whatever is input."""
    xvals = np.asarray(xvals)
    res = np.asarray(res)
    if ham_input: 
        xvals = hamm_to_cosine(xvals, n)
    zeros_in_res = False
    # need to remove any zeros for this calculation. 
    if res[-1] == 0.0:
        print("res equals 0, problem for the log. Removing from the equation here.")
        mask = res!=0.0
        num_zeros = (res==0.0).sum()
        res = res[mask]
        xvals = xvals[mask]
        zeros_in_res = True
    yvals = np.log(np.asarray(res))
    # log linear regression closed form solution. 
    beta = np.cov(xvals, yvals)[0][1] / np.var(xvals)
    bias = np.mean(yvals) - beta*np.mean(xvals)
    #mse between res and beta res: 
    #print('Beta Fit MSE:',np.sum((res-np.exp(beta*xvals)+bias)**2)/len(res) )
    
    if return_bias: 
        return beta, bias
    else: 
        return beta
    
def fit_softmax_backprop(n, dvals, targets, lr=0.3, niters=5000, ham_input=False, plot_losses=True):
    """
    Learns an approximation to the circle intersection that is normalized. Ie fits a softmax function. This is unrealistic in 
    that it overfits to the softmax rather than the exponential approximation where the softmax is conditioned upon the number of inputs in the normalizing constant. But is still 
    interesting to analyze for what a perfect Beta fit to a particular softmax would be. 
    """
    # 
    targets = torch.Tensor(targets/sum(targets))
    if ham_input: 
        xvals = torch.Tensor( hamm_to_cosine(dvals, n) )
    else: 
        xvals = torch.Tensor(dvals)
        
    beta = torch.nn.Parameter(torch.Tensor(np.random.uniform(1,30, 1)), requires_grad=True)
    optimizer = optim.Adam([beta], lr=lr)
    
    losses = []
    for i in range(niters):
        # training loop:
        optimizer.zero_grad()   # zero the gradient buffers
        preds = F.softmax(beta*xvals)
        loss = ((targets-preds)**2).sum() / len(dvals)
        loss.backward()
        optimizer.step() 
        losses.append(loss.item())
    if plot_losses:
        plt.figure()
        plt.plot(losses)
        plt.title("Losses during learning")
        plt.show()
    print("final loss", loss.item())
    
    return beta.item()
    
    
def integral_func(phi, th1, n):
    """ Used in computing the continuous hypersphere cap intersection below. """
    return np.sin(phi)**(n-2) * scipy.special.betainc( (n-2)/2 , 1/2, 1-( (np.tan(th1))/(np.tan(phi)) )**2 )

def log_J_n(th1, th2, r, n):
    """ Used in computing the continuous hypersphere cap intersection below. """
    integral = quad(integral_func, th1, th2, args=(th1, n) )[0]
    #print(np.log(np.pi**( (n-1) /2) ) , scipy.special.loggamma( (n-1) /2), np.log(r**(n-1)), np.log(integral ))
    return np.log(np.pi**( (n-1) /2) ) - scipy.special.loggamma( (n-1) /2) + np.log(r**(n-1)) + np.log(integral )
    
def cap_intersection(n, cs_dvs, hamm_dist, r, rad=1, 
                         return_log=False, ham_input = False, print_oobs=False):
    """ 
     Computes the continuous hypersphere cap intersection. 
     Does all compute in log space for numerical stability, option to return 
     log results or not. 
    """
    if r is not None: 
        if type(r) != int: 
            r = np.round(r) # number of neurons
        r = float(r)
    else: 
        r = 1.0
    #size of total space
    log_total_space = log_hypersphere_sa(n,rad)

    if ham_input: 
        cs_dvs = hamm_to_cosine(cs_dvs)
    c_dist  = hamm_to_cosine(hamm_dist,n)
    t1 = t2 = np.arccos(c_dist)
    log_inters = []
    for cs_dv in cs_dvs:
        tv = np.arccos(cs_dv)
        if tv>=t1+t2 or t1+t2>(2*np.pi)-tv:
            if print_oobs:
                print("out of equation bounds", cs_dv)
            log_inters.append(np.nan)
            continue
        tmin = np.arctan( (np.cos(t1)/(np.cos(t2)*np.sin(tv))) - (1/np.tan(tv)) )
        assert np.round(tmin,5) == np.round(tv-tmin,5)
        assert np.round(t2,5)==np.round(t1,5)
        
        log_inters.append(2+log_J_n(tmin, t2, rad, n) )#np.log( np.exp(log_J_n(tmin, t2, rad, n))+np.exp(log_J_n(tv-tmin, t1, rad, n))) )
    log_inters = np.asarray(log_inters)
    #print(log_inters, log_total_space)
    #print(np.exp(log_inters), np.exp(log_total_space))
    log_fracs_of_space = log_inters - log_total_space 
    log_num_expected_neurons = log_fracs_of_space + np.log(r)
    if return_log:
        # have not removed the nans either
        log_num_expected_neurons = np.nan_to_num(log_num_expected_neurons, nan=-1e+30)
        return log_num_expected_neurons
    else: 
        num_expected_neurons = np.exp(log_num_expected_neurons)
        num_expected_neurons = np.nan_to_num(num_expected_neurons, nan=0.0)
        return num_expected_neurons
    
def log_hypersphere_sa(n, rad=1):
    # n dim hypersphere SA
    # https://en.wikipedia.org/wiki/Unit_sphere
    # assuming L2 norm with r=1!
    return np.log(2* (np.pi**(n/2) ) ) - scipy.special.loggamma(n/2)  + np.log(rad**(n-1)) 

def hypersphere_v(n, r):
    return (np.pi**(n/2) )/(scipy.special.gamma((n+1)/2) )*(r**n)
    
def expected_intersection_lune(n, dvals, hamm_dist, r):
    # This equation gives the same results as the one we derive and present in the paper. It was introduced in the SDM book and runs a bit faster. 
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
    
    if r is not None: 
        r = int(r)
        perc_addresses_w_neurons = r/(2**n) 
    else: 
        perc_addresses_w_neurons = 1.0
    
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
        
    res = np.asarray(res)
    return res

def expected_intersection_interpretable(n, dvals, hamm_dist, r, weight_type=None):
    
    if r is None: 
        r = 1.0
    
    perc_addresses_w_neurons = np.log(float(r)) - np.log(2.0**n)
    res = []
    for dval in dvals:

        possible_addresses = 0
        for a in np.arange(n-hamm_dist-(dval//2),n+0.1-dval):

            # solve just for b then c is determined. 
            bvals = np.arange(np.maximum(0,n-hamm_dist-a), dval-(n-hamm_dist-a)+0.1) # +0.1 to ensure that the value here is represented.
            #print(a, 'b values', bvals)
            if len(bvals)==0:
                continue
                
            if weight_type == "Linear":
                # linear weighting from the read and write operations. 
                weighting = ((a+bvals)/n) * ( (a+(dval-bvals))/n )
            if weight_type == "Expo":
                # linear weighting from the read and write operations. 
                weighting = np.exp(-0.01*(n-(a+bvals))) * np.exp(-0.01*(n-(a+(dval-bvals))))
            elif not weight_type: 
                weighting = 1

            possible_addresses += comb(n-dval,a)*(weighting*comb(dval,bvals)).sum()
        expected_intersect = perc_addresses_w_neurons + np.log(possible_addresses)
        res.append(np.exp(expected_intersect))
    return np.asarray(res)

    
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
    
    
def plot_line(x, y, label_prefix, label_val, norm=True):
    label = label_prefix
    if label_val: 
        label +=str(label_val)
    if norm: 
        y = y/sum(y)
    plt.plot(x, y, label=label)

def label_plot(title, norm=True, directory="figures/Jaeckel_Analysis/", save_name=None):
    plt.legend()
    plt.title(title)
    plt.xlabel('Hamming Distance Between Pattern and Query')
    if norm: 
        plt.ylabel('Normalized overlap weights')
    else: 
        plt.ylabel('Expected neurons in intersection')
    if save_name: 
        plt.gcf().savefig(directory+save_name+'.png', dpi=250)
    plt.show()
    
    
def SDM_Interpretable(params, dvals, thresholds, title=None, label_prefix='ham='):
    """Same as the SDM lune equation in results. Equation was inspired by Jaeckel's SDM Hyperplane but applied to the SDM setting with binary vectors and optimized by working out lower and upper bounds to avoid using a CSP. This equation is much more interpretable than the Lune one used in the SDM Appendix B.
    
    See paper for the constraints and bounds explained."""
    
    perc_addresses_w_neurons = np.log(params.r) - np.log(2.0**params.n)

    for thresh in thresholds:
        res = []
        for dval in dvals:

            possible_addresses = 0
            #print('range of a vals', np.arange(params.n-thresh-(dval//2),params.n+1-dval))
            for a in np.arange(params.n-thresh-(dval//2),params.n+0.1-dval):

                # solve just for b then c is determined. 
                bvals = np.arange(np.maximum(0,params.n-thresh-a), dval-(params.n-thresh-a)+0.1) # +0.1 to ensure that the value here is represented.
                #print(a, 'b values', bvals)
                if len(bvals)==0:
                    continue

                possible_addresses += comb(params.n-dval,a)*comb(dval,bvals).sum()
            expected_intersect = perc_addresses_w_neurons + np.log(possible_addresses)
            res.append(np.exp(expexcted_intersect))
        res =np.asarray(res)
        plot_line(dvals, res, label_prefix, thresh, params.norm)
        
        if params.fit_beta_and_plot_attention:
            fit_beta_res, beta = fit_beta_regression(params.n, dvals, res)
            plot_line(dvals, fit_beta_res, 'fit_beta | '+label_prefix, thresh, params.norm)
    
    if title: # else can call "label plot separately"
        label_plot(title, params.norm)
    return res

def SDM_lune(params, dvals, title=None, label_prefix='ham='):
    """Exact calculation for SDM circle intersection. For some reason mine is a slight upper bound on the results found in the book. Uses a proof from Appendix B of the SDM book (Kanerva, 1988). Difference is neglible when norm=True."""

    res = expected_intersection_lune(params.n, dvals, params.hamm_dist, params.r )
    
    if params.plot_lines:
        plot_line(dvals, res, label_prefix, params.hamm_dist, params.norm)
    
    if params.fit_beta_and_plot_attention:
        fit_beta_res, beta = fit_beta_regression(params.n, dvals, res)
        plot_line(dvals, fit_beta_res, 'fit_beta | '+label_prefix, params.hamm_dist, params.norm)

    if title: # else can call "label plot separately"
        label_plot(title, params.norm)
        
    return res

def f(x, c_p):
    """This is used in the continuous approximation to the circle intersection derived in Appendix B of the SDM book that needs to be numerically integrated. It is less accurate than the exact equation we outline in the paper and use for our circle intersection computations in all figures and analyses unless otherwise noted."""
    return 1/(2*np.pi*np.sqrt(x*(1-x)))*np.exp(-0.5*(c_p**2/(1-x)))

def expected_intersection_continuous(n, dvals, hamm_dist, r, hard_mem_places):
    """
    Computes the fraction of the space that exists in the circle intersection using the continuous approximation to the Lune equation. NOTE THIS IS AN APPROXIMATION FROM THE SDM BOOK THAT IS INACCURATE. 
    
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
