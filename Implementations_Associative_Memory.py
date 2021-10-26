"""
Author: Trenton Bricken @trentbrick

Implements the following Associative Memory Algorithms: 

* Neuron Based SDM - Biologically Plausible. NB. This algorithm takes a long time to run because of the number of Hamming distance computations that must be done (m patterns * r neurons to initialize memory value vectors), then m operations for every query update. It remains slow and memory intensive even with the parallelized version used here. 

* Pattern Based SDM with Infinite Neurons - Assumes there are $r=2^n$ neurons in the space. Uses the circle intersection equation that should approximate a binary version of Attention well. 

* Pattern Based SDM Limited Neurons - Same as the above but enforces that there are in fact only $r$ neurons in the space. 

* Continuous SDM - Uses the continuous version of SDM where it computes the intersection of cones on a hypersphere. 

* Binary SDM Attention - Uses Attention equation with the Softmax and a learnt beta but with binary vectors and Hamming distance. Beta is learnt using a linear regression to approximate the circle intersection.

* SDM Attention - Uses Attention equation with the Softmax and a learnt beta with continuous vectors and L2 norm. Beta is learnt using a linear regression to approximate the circle intersection.

* Transformer Attention - Using continuous values that are L2 normalized. Beta = 1/\sqrt{n} as in the original Transformer.

* Hopfield Network - Not used in the paper.

"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import binom
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import ray
import psutil
#import tensorflow as tf
num_cpus = psutil.cpu_count(logical=False)
from SDM_Circ_Inter_Funcs import *
import torch.optim as optim
import torch.nn.functional as F

def softmax(x, beta):
    assert len(x.shape) <3, 'this softmax can currently only handle vectors'
    x = x * beta
    return np.exp(x)/np.exp(x).sum()

def d(X, Y, n):
    # computes Hamming distance. 
    # expects vectors to be compared as columns
    return cdist(X.T.astype(bool), Y.T, metric='hamming')*n

@ray.remote
def d_parallel_comp(X, Y, n):
    # parallelized computation of Hamming distance. 
    return cdist(X.T.astype(bool), Y.T, metric='hamming')*n

def d_parallel(X, Y, n):
    # expects vectors to be compared as columns
    split_size = X.shape[1]//num_cpus
    batches = [ X[:,i*split_size:(i+1)*split_size] for i in range(num_cpus-1)]
    batches.append(X[:,((num_cpus-1)*split_size):])

    result_ids = [ d_parallel_comp.remote(batches[i], Y, n) for i in range(num_cpus) ]
    results = ray.get(result_ids)
    results = np.concatenate(results, axis=0)
    
    return results

def parallel_pool_caller(job_deets):
    # used with ray to parallelize jobs. 
    fn, params, raw_patterns_flat, target_patterns, dataset_num, queries, plot, image = job_deets
    #print('running a job, dataset num is', dataset_num, fn)
    n, r, m, h_dist, max_iterations, image_len, print_each_result = params
    return (fn(params, raw_patterns_flat, queries, plot=plot, image=image), target_patterns, fn, h_dist, dataset_num)


def binary_data_prep(params, raw_flat_patterns, queries):
    n, r, m, h_dist, max_iterations, image_len, print_each_result = params
    # binarize the queries and patterns:
    #assert (raw_flat_patterns<0.0).sum()==0, "Some pattern inputs are negative! Can't binarize correctly."
    #assert (queries<0.0).sum()==0, "Some query inputs are negative! Can't binarize correctly."
    P_a = (raw_flat_patterns[:, :m]>0.0).astype(int)
    queries = (queries>0.0).astype(int)
    return P_a, queries

def b(X, h_dist): # returns if a value is below a given Hamming distance
    return (X<=h_dist).astype(bool)

def g(X): # majority update
    return (X>0.5).astype(bool)

def g_hop(X): # majority update for bipolar values
    return ((X>0).astype(int)*2)-1

def perturb_queries(queries, P, norm_keys):
    # Uses Pytorch to take a pattern and provide a desired perturbation to it.
    # Code taken from:
    # https://stackoverflow.com/questions/52916699/create-random-vector-given-cosine-similarity
    bs = queries.shape[1]
    if P.fix_perturb:
        cs_perturbs = torch.ones((bs), device=P.device) * (1-((2*P.perturb)/P.img_flat_len))
    else: 
        rand_perturbs = torch.rand((bs), device=P.device)*(P.max_perturb-P.min_perturb)+P.min_perturb
        cs_perturbs = torch.ones((bs), device=P.device) * (1-((2*rand_perturbs)/P.img_flat_len))
    us = queries/torch.norm(queries, dim=0, keepdim=True)
    rand_vects = torch.rand(queries.shape, device=P.device)*2 - 1
    target_perps = rand_vects - (rand_vects*us).sum(dim=0)*us
    target_perps = target_perps / torch.norm(target_perps, dim=0, keepdim=True)
    queries = cs_perturbs*us + torch.sqrt(1-cs_perturbs**2)*target_perps

    # check that target is still the closest: 
    vs,_ = torch.max(norm_keys@queries, dim=0) 
    isclosests = vs <= cs_perturbs+0.00001
    if P.check_for_closer_images:
        assert isclosests.sum() == bs, "another image is closer!"
    return queries, isclosests

def convergence_loop(W_k, pKs, queries, norm_keys, keys, P, 
                     cached_intersects=None, 
                     continuous_cached_intersects=None ):
    # Convergence loop for Pytorch implementation of the projection matrices. 
    queries, isclosests = perturb_queries(queries, P, norm_keys)
    
    bits_changed_sum = 100000 # some large number
    i = 0
    # will keep going as long as none of them are satisfied. 
    while bits_changed_sum > 0 and i<P.max_conv_steps: 
        #if renorm_query:
        #    query = query/torch.norm(query, dim=0,keepdim=True)
        pQs = W_k@queries
        # L2 norm all
        pQs = pQs/torch.norm(pQs, dim=0, keepdim=True)
        # columnwise cosine similarities
        dotps = pKs.T@pQs

        # either apply softmax operation or map cosine similarity to hamming distance. and then circle intersection
        if P.use_softmax: 
            new_queries = keys.T@F.softmax( P.beta*dotps, dim=0 )
        else: 
            if P.use_continuous: 
                # need to map dotps to the nearest precomputed cosine value. 
                nearest_inds = (P.cont_cache_resolution/2*(dotps + 1)).type(torch.long)-1
                intersects = continuous_cached_intersects[nearest_inds]
            else: 
                dvs = cosine_to_hamm(dotps, P.n)
                dvs = torch.round(dvs).type(torch.long)
                intersects = cached_intersects[dvs]
            if P.r is not None: 
                # converting to integer as dealing with discrete neurons here. 
                decimals = intersects%1
                one_more_neuron_sample = (torch.rand(intersects.shape, device=P.device)<decimals).type(torch.float)
                intersects = intersects.type(torch.int)+one_more_neuron_sample
            res = keys.T@intersects 
            new_queries = res/intersects.sum(0)
            
            # any intersect that is 0 needs to be ignored to avoid nan errors. the query should just not update at all
            no_intersects = intersects.sum(0)==0.0
            new_queries[:, no_intersects] = queries[:, no_intersects]

        bits_changed_sum = P.img_flat_len-torch.logical_and(queries-P.epsilon < new_queries , 
                             queries+P.epsilon > new_queries).sum()
        queries=new_queries
        i+=1
        
    return queries, isclosests, i

def train_projection(W_k, P, trainloader, keys, norm_keys):
    ''' 
    Optimization to learn the key projection matrix.

    args:: 
    W_k = Key projection matrix
    P = training parameters
    keys = all of the patterns
    norm_keys = all of the patterns, normalized. 

    returns:: 
    W_k = Key projection matrix Torch.Tensor
    '''

    optimizer = optim.Adam([W_k], lr=P.lr)
    losses = []

    for e in range(P.proj_epochs): # to train the projection matrix
        
        perc_recons = []
        closests = []

        for batch_ind, queries in enumerate(trainloader): 
            # get a minibatch
            bs = len(queries)
            queries = queries.to(P.device).T # making the queries into columns. 
            targets = torch.clone(queries)

            # training loop:
            optimizer.zero_grad()   # zero the gradient buffers

            #project keys
            pKs = W_k@keys.T
            pKs = pKs/torch.norm(pKs, dim=0,keepdim=True)

            queries, isclosests, nsteps = convergence_loop(W_k, pKs, queries, norm_keys, keys, P)
            
            # reconstruction loss after convergence iterations. 
            loss = ((queries - targets)**2).sum()

            perc_recon =  torch.logical_and(targets-P.epsilon < queries , 
                                     targets+P.epsilon > queries).sum(0).cpu().detach().numpy() / P.img_flat_len
            perc_recons+=list(perc_recon)
            closests+=list(isclosests.cpu().detach().numpy())

            loss /= bs
            loss.backward()
            optimizer.step() 

            losses.append(loss.item())
            #imshow(target.detach().view(nchannels,32,32))
            #imshow(query.detach().view(nchannels,32,32))
            if P.plot_batches: 
                if P.batch_ind%P.plot_batches==0:
                    plt.plot(losses)
                    plt.show()

                    plt.imshow(W_k[:100,:100].cpu().detach().numpy())
                    plt.show()
                    print( "All -- Average percentage reconstructed", sum(perc_recons)/len(perc_recons) )

        print("epoch", e)
        if P.plot_epochs: 
            plt.plot(losses)
            plt.show()
            plt.imshow(W_k[:100,:100].cpu().detach().numpy())
            plt.show()

            print( "All -- Average percentage reconstructed", sum(perc_recons)/len(perc_recons) )
            #print(perc_recons)
            perc_recons = np.asarray(perc_recons)
            print( "Closests -- Average percentage reconstructed", sum(perc_recons[closests])/len(perc_recons[closests]) )
            #print(closests)
            print("number not closest", len(closests) - sum(closests))
            print('=======')
        
    # do a final plot of this training run. 
    print("epoch", e) 
    plt.plot(losses)
    plt.show()
    plt.imshow(W_k[:100,:100].cpu().detach().numpy())
    plt.show()

    print( "All -- Average percentage reconstructed", sum(perc_recons)/len(perc_recons) )
    perc_recons = np.asarray(perc_recons)
    print( "Closests -- Average percentage reconstructed", sum(perc_recons[closests])/len(perc_recons[closests]) )
    print("number not closest", len(closests) - sum(closests))
    print('=======')
        
    return W_k

def test_projection(pert_dists, W_k, P, trainloader, 
                     testloader, keys, norm_keys,
                     tkeys, tnorm_keys, cached_intersects,
                     continuous_cached_intersects, plot_results=True):
    """
    Testing the trained projection matrix on all train and test data 
    using all three algorithms: 
    Binary, Continuous and Softmax. 
    
    args::
    pert_dists = list of perturbation Hamming distances 
    W_k = trained projection matrix
    P = testing parameters
    trainloader = Pytorch Dataloader for training data. 
    testloader = testing data.
    keys = train data keys
    norm_keys = train data L2 norm keys
    tkeys = test keys
    tnorm_keys 
    cached_intersects = cached SDM binary circle intersection values
    continuous_cached_intersects = cached SDM continuous circle intersect values
    
    returns::
    pert_res = percentage reconstructed, uses an epsilon similarity threshold. Less accurate 
    than using cosine similarity. 
    cos_res = cosine similarity of converged query to target pattern
    """
    pert_res = {'train':[], 'test':[]}
    cos_res = {'train':[], 'test':[]}
    
    for perturb in pert_dists:
        P.perturb=perturb
        if plot_results:
            print("!!! Perturb is:", perturb)
        with torch.no_grad(): 
            for form, loader in zip(['train','test'],[trainloader, testloader]):
                if plot_results:
                    print('============',form,'===========')
                losses = []
                perc_recons = []
                cosine_sims = []
                closests = []

                if form =='train':
                    #project keys
                    pKs = W_k@keys.T
                else: 
                    pKs = W_k@tkeys.T
                pKs = pKs/torch.norm(pKs, dim=0,keepdim=True)

                for lind, queries in enumerate(loader):
                    #print('batch index', lind)
                    # get a minibatch
                    bs = len(queries)
                    queries = queries.to(P.device).T # making the queries into columns. 
                    targets = torch.clone(queries)

                    if form =='train':
                        queries, isclosests, nsteps = convergence_loop(W_k, pKs, 
                                                                       queries, norm_keys, 
                                                                       keys, P, cached_intersects,
                                                                       continuous_cached_intersects)
                    else: 
                        queries, isclosests, nsteps =convergence_loop(W_k, pKs, 
                                                                       queries, tnorm_keys, 
                                                                       tkeys, P, cached_intersects,
                                                                       continuous_cached_intersects)
                    #print('nsteps', nsteps)
                    # reconstruction loss after convergence iterations. 
                    loss = ((queries - targets)**2).sum()
                    cosine_sim = (queries/torch.norm(queries, dim=0,keepdim=True)*targets/torch.norm(targets, dim=0,keepdim=True)).sum(0).cpu().detach().numpy()
                    
                    # NB using test epsilon here!!! 
                    perc_recon =  torch.logical_and(targets-P.test_epsilon < queries , 
                                     targets+P.test_epsilon > queries).sum(0).cpu().detach().numpy() / P.img_flat_len  
                    losses.append(loss.item())
                    perc_recons+=list(perc_recon)
                    cosine_sims += list(cosine_sim)
                    closests+=list(isclosests.cpu().detach().numpy())
                    
                perc_recons = np.asarray(perc_recons)
                cosine_sims = np.asarray(cosine_sims)
                pert_res[form].append(perc_recons)#sum(perc_recons)/len(perc_recons))
                cos_res[form].append(cosine_sims)#sum(cosine_sims)/len(cosine_sims))
                if plot_results:
                    plt.hist(losses, bins=100)
                    plt.title("Losses")
                    plt.show()
                    plt.hist(cosine_sims, bins=100)
                    plt.title("Cosine Sims")
                    plt.show()
                    print( "All -- Average cosine sim", sum(cosine_sims)/len(cosine_sims) )
                    print( "Closests -- Average cosine sim", sum(cosine_sims[closests])/len(cosine_sims[closests]) )
                    print("number not closest", len(closests) - sum(closests))
    return pert_res, cos_res

def Neuron_SDM(params, raw_flat_patterns, queries_tuple, plot=True, image=False):
    
    """
    The original SDM implementation that assumes the existence of neurons in the space. 
    
    """
    
    
    queries = queries_tuple[0] # hamming distance version
    
    # patterns should be 2D and the columns should be unique patterns. 
    assert len(raw_flat_patterns.shape) ==2, 'patterns are wrong dimenions.'
    
    n, r, m, h_dist, max_iterations, image_len, print_each_result = params
    
    #make the raw patterns binary:
    P_a, init_queries = binary_data_prep(params, raw_flat_patterns, queries)
    # assuming autoassociative for now:
    P_p = P_a
    
    # generate random neuron addresses:
    X_a = binom.rvs(1, 0.5, size=(n,r))
    
    X_a_P_a_interactions = b(d(P_a, X_a,n), h_dist)
    X_v = P_p@X_a_P_a_interactions
    
    h_v = (np.ones(m)@X_a_P_a_interactions ).reshape(1,-1)
    
    if plot: 
        plt.hist(X_v.flatten(),bins=50)
        plt.title('values of all the counters')
        plt.show()
        
        plt.hist(h_v[0,:], bins=50)
        plt.title("number of connections each neuron has.")
        plt.show()

    converged_queries = []
    for init_query in init_queries: 
        query = np.copy(init_query)
        
        # positive number to trick the while loop
        bits_changed = 100
        i = 0 # iteration tracker
        
        while bits_changed > 0 and i<max_iterations:

            query_neuron_inter = b(d(X_a, query,n), h_dist)

            pre_g = ( X_v@query_neuron_inter ) / ( h_v @ query_neuron_inter )
            new_query = g( pre_g )
            
            bits_changed = (new_query!=query).sum()
            query = new_query
            i+=1
            
            if plot and image: 
                plt.imshow(pre_g.reshape(image_len,image_len))
                plt.show()

                plt.imshow(query.reshape(image_len,image_len))
                plt.title('OG SDM result after: '+str(i)+' iterations')
                plt.show()

        if image: 
            plt.imshow(query.reshape(image_len,image_len))
            plt.title('OG SDM result after: '+str(i)+' iterations')
            plt.show()
            
        converged_queries.append(query)

        if print_each_result:
            print('Percentage pixels that agree with (noisy) query', (init_query==query).sum()/n)
    
    return converged_queries
    
def fit_beta(n,r,h_dist, plot=False,hard_mem_places = False, return_bias=False):
    """
    Fit a beta coefficient to a given SDM binary circle intersection defined by its dimensional n, number of neurons r, and the hamming radius h_dist. 
    """
    res = []
    drange = np.arange(0,h_dist) # fit to the chopped index which is at hamming distance.
    res = expected_intersection_lune(n, drange, h_dist, r)
    res = np.asarray(res)

    if plot: 
        plt.plot(drange, res/res.sum(), label='Truth')
        
    out = fit_beta_regression(n, drange, res, return_bias=return_bias )
    if return_bias: 
        beta = out[0]
        bias = out[1]
    else: 
        beta = out
    
    #print('beta that has been fit', beta)

    if plot: 
        plt.plot(drange, softmax(fit_beta_res,beta), label="Softmax w/ learned "+ r"$\beta$")
        #plt.plot(drange, softmax(n-drange, 1/np.sqrt(n)), label='Softmax Attention '+r'$\beta=\dfrac{1}{\sqrt{n}}$')
        plt.legend()
        print('learned softmax beta vs attention beta:', beta, 1/np.sqrt(n))
        plt.xlabel('Hamming Distance between circles')
        plt.ylabel('Normalized Weight')
        plt.show()
    
    if return_bias: 
        return beta, bias
    return beta
    
def SDM_Attention(params, raw_flat_patterns, queries_tuple, renorm_query =True, plot=True, image=False, fit_beta_param=True):
    """
    Using Attention operations with the softmax and continuous vectors but fitting 
    beta to the equivalent SDM circle intersection. 
    """
    
    # auto associative attention. 
    # this is the same as 
    # MCHN/Continuous Softmax Auto-Associative SDM
    
    queries = queries_tuple[1]
    
    # patterns should be 2D and the columns should be unique patterns. 
    assert len(raw_flat_patterns.shape) ==2, 'patterns are wrong dimenions.'
    
    n, r, m, h_dist, max_iterations, image_len, print_each_result = params
    # overriding max iterations here to be 200
    #max_iterations = 100
    epsilon = 0.05 # removes any rounding error in computing the similarity.
    P_a_cont = raw_flat_patterns[:, :m]
    
    # need to normalize all of the patterns and the queries
    P_a_cont = P_a_cont / np.linalg.norm(P_a_cont, axis=0)
    
    if fit_beta_param:
        beta = fit_beta(n,r,h_dist, plot=plot,hard_mem_places = False)
    else:
        beta = 1/np.sqrt(n)
        
    #print('SDM attention beta being used:', beta)
    
    #if print_each_result:
    #    print('real beta being used is *n:', beta*n)
    
    converged_queries = []
    for query in queries:
    
        init_query = query/np.linalg.norm(query)
        query = np.copy(init_query)

        bits_changed = 100
        i = 0
        while bits_changed > 0 and i<max_iterations:
            new_query = P_a_cont@softmax( P_a_cont.T@query, beta )
            bits_agree = np.logical_and(query-epsilon < new_query , 
                                 query+epsilon > new_query).sum()
            bits_changed = n-bits_agree #(new_query!=query).sum()
            query=new_query
            i+=1

            if renorm_query:
                query = query/np.linalg.norm(query)

            if plot and image: 
                plt.imshow(query.reshape(image_len,image_len))
                plt.title('Autoassociative Attention result after: '+str(i)+' iterations')
                plt.show()

        if image: 
            plt.imshow(query.reshape(image_len,image_len))
            plt.title('Autoassociative Attention result after: '+str(i)+' iterations')
            plt.show()

        #print("Attention number of iterations till convergence", i)
        if print_each_result:
            print('Percentage pixels that agree with (noisy) query', np.logical_and(init_query-epsilon < query , init_query+epsilon > query).sum()/n)
            print('epsilon value being used for this equality check =', epsilon)

        converged_queries.append(query)
            
    return converged_queries
    
    
def Transformer_Attention(params, raw_flat_patterns, queries_tuple, renorm_query =True, plot=True, image=False):
    """ Using Attention but not fitting the Beta coefficient, and having it be
    fixed at 1/sqrt(n), this is not a meaningful algorithm in so much as 
    we do not learn our 
    vector norms which would be required to create effective beta values. 
    """
    # calls SDM Attention code just does not fit the Beta Value. 
    return SDM_Attention(params, raw_flat_patterns, queries_tuple, renorm_query =renorm_query, plot=plot, image=image, fit_beta_param=False)
    
def Binary_SDM_Attention(params, raw_flat_patterns, queries_tuple, plot=True, image=False):
    # same as above but binary instead of continuous
    
    queries = queries_tuple[0]
    
    # patterns should be 2D and the columns should be unique patterns. 
    assert len(raw_flat_patterns.shape) ==2, 'patterns are wrong dimenions.'
    
    n, r, m, h_dist, max_iterations, image_len, print_each_result = params
    
    #make the raw patterns binary:
    P_a, init_queries = binary_data_prep(params, raw_flat_patterns, queries)
    # assuming autoassociative for now:
    P_p = P_a
    
    beta = fit_beta(n,r,h_dist, plot=plot,hard_mem_places = False)
    
    # binarize the queries:
    converged_queries = []
    for init_query in init_queries:
        query = np.copy(init_query)
        
        bits_changed = 100
        i = 0
        while bits_changed > 0 and i<max_iterations:

            pre_bin = P_p@softmax( 1-(2*d( P_a, query, n))/n , beta )
            new_query = g(pre_bin)
            
            bits_changed = (new_query!=query).sum()
            query=new_query
            i+=1

            if plot and image: 
                plt.imshow(pre_bin.reshape(image_len,image_len))
                plt.title('Pre Binarization threshold')
                plt.show()
                plt.imshow(query.reshape(image_len,image_len))
                plt.title('Binary Autoassociative Attention result after: '+str(i)+' iterations')
                plt.show()
        #print("Binary Attention number of iterations till convergence", i)
        if image:
            plt.imshow(query.reshape(image_len,image_len))
            plt.title('Binary Autoassociative Attention result after: '+str(i)+' iterations')
            plt.show()

        if print_each_result:
            print('Percentage pixels that agree with (noisy) query', (init_query==query).sum()/n)
    
        converged_queries.append(query)
    
    return converged_queries
    
def update_circle_query(params, P_a, q, cached_intersects, enforce_num_neurons, 
                            continuous=False, resolution=10000):
    """
    Numpy updating of the query using cached circle intersection values. 
    This is called by Continuous_SDM, Pattern_SDM and NeuronLimited_Pattern_SDM.
    
    """
    
    n, r, m, h_dist, max_iterations, image_len, print_each_result = params 
    if continuous: 
        dotps = P_a.T@q
        cache_inds = (resolution/2*(dotps + 1))-1
    else: 
        # get query pattern hamming distances:
        cache_inds = d(P_a, q, n)
        
    intersects = cached_intersects[cache_inds.astype(int)]
    if enforce_num_neurons: 
        if continuous: 
            decimals = np.exp(intersects)%1
            one_more_neuron_sample = (np.random.uniform(size=(len(intersects),1))<decimals).astype(float)
            intersects = np.log(np.floor(np.exp(intersects))+one_more_neuron_sample) # need to enforce hard cutoff. 

        else: 
            decimals = intersects%1
            one_more_neuron_sample = (np.random.uniform(size=(len(intersects),1))<decimals).astype(float)
            intersects = np.floor(intersects)+one_more_neuron_sample # need to enforce hard cutoff. 
    # get pattern weightings
    if continuous: 
        # intersects are in log space. 
        unnorm = None #TODO: implement this plotting function if needed.
        res = (np.exp(intersects.T) * P_a).sum(axis=1)/np.exp(intersects).sum()
        #print("res", res)
    else: 
        res = (intersects.T * P_a).sum(axis=1) # broadcasting
        unnorm = np.copy(res)
        res = res/intersects.sum()
        # binarizing.
        res = g(res)
    return res, unnorm
    
def Pattern_SDM(params, raw_flat_patterns, queries_tuple, plot=True, image=False, enforce_num_neurons=False):
    """
    Computes the expected circle intersection rather than actual neuron positions. 
    If enforce_num_neurons = True. It will convert the circle intersection to an integer
    number of neurons.
    
    """
    queries = queries_tuple[0]
    # patterns should be 2D and the columns should be unique patterns. 
    assert len(raw_flat_patterns.shape) ==2, 'patterns are wrong dimenions.'
    
    n, r, m, h_dist, max_iterations, image_len, print_each_result = params
                       
    P_a, init_queries = binary_data_prep(params, raw_flat_patterns, queries)
    # assuming autoassociative for now:
    P_p = P_a
    
    # caching of the circle intersections: 
    all_dvs = np.arange(0,n+1)
    if enforce_num_neurons: 
        cached_intersects = expected_intersection_lune(n, all_dvs, h_dist, r)
    else: 
        cached_intersects = expected_intersection_lune(n, all_dvs, h_dist, None)
    
    converged_queries = []
    for query_ind, init_query in enumerate(init_queries):
        #print("processing query number:", query_ind)
        query = np.copy(init_query)

        bits_changed = 100 # placeholder to be able to track convergence.
        i = 0
        while bits_changed > 0 and i<max_iterations:

            new_query, pre_bin = update_circle_query(params, P_a, query, cached_intersects, enforce_num_neurons, continuous=False)
            new_query = new_query.reshape(-1,1)
            
            bits_changed = (new_query!=query).sum()
            query=new_query
            i+=1

            if plot and image: 
                plt.imshow(pre_bin.reshape(image_len,image_len))
                plt.title('Pre Binarization threshold')
                plt.show()
                plt.imshow(query.reshape(image_len,image_len))
                plt.title('Pattern_SDM result after: '+str(i)+' iterations')
                plt.show()

        if image: 
            plt.imshow(query.reshape(image_len,image_len))
            plt.title('Pattern_SDM result after: '+str(i)+' iterations')
            plt.show()

        if print_each_result: 
            print('Percentage pixels that agree with (noisy) query', (init_query==query).sum()/n)
            
        #print("number of steps taken", i)
    
        converged_queries.append(query)
    
    return converged_queries


def Pattern_SDM_NeuronLimit(params, raw_flat_patterns, queries_tuple, plot=True, image=False):
    """
    Calls Pattern SDM but enforces a finite number of neurons. 
    """
    return Pattern_SDM(params, raw_flat_patterns, queries_tuple, plot=plot, image=image, enforce_num_neurons=True)
    
def Continuous_SDM(params, raw_flat_patterns, queries_tuple, plot=True, image=False, enforce_num_neurons=False, renorm_query =True):
    """
    OG SDM but with circle intersections that are continuous using the Appendix equation where we have the intersection of two caps on a hypersphere. Expected intersections so dont model neurons. 
    """
    queries = queries_tuple[1] # continuous version. 
    # patterns should be 2D and the columns should be unique patterns. 
    assert len(raw_flat_patterns.shape) ==2, 'patterns are wrong dimenions.'
    
    n, r, m, h_dist, max_iterations, image_len, print_each_result = params
    
    epsilon = 0.05 # removes any rounding error in computing the similarity.
    P_a_cont = raw_flat_patterns[:, :m]
    
    # need to normalize all of the patterns and the queries
    P_a_cont = P_a_cont / np.linalg.norm(P_a_cont, axis=0)
    # caching of the circle intersections: 
    all_dvs = np.arange(0,n+1)
    
    resolution = 10000
    cs_intervals = np.linspace(-1,1,resolution).astype(float)
    cs_intervals[-1] = cs_intervals[-1] - 1e-15

    if enforce_num_neurons: 
        log_continuous_cached_intersects = cap_intersection(n, cs_intervals, h_dist, r, 
                                                        return_log=True,
                                                        ham_input=False, print_oobs=False)
    else: 
        log_continuous_cached_intersects = cap_intersection(n, cs_intervals, h_dist, None, 
                                                        return_log=True,
                                                        ham_input=False, print_oobs=False)
    
    converged_queries = []
    for query_ind, init_query in enumerate(queries):
        #print("processing query number:", query_ind)
        query = np.copy(init_query)

        bits_changed = 100 # placeholder to be able to track convergence.
        i = 0
        while bits_changed > 0 and i<max_iterations:

            new_query, pre_bin = update_circle_query(params, P_a_cont, query, log_continuous_cached_intersects,  
                                                         enforce_num_neurons, 
                                                         continuous=True, resolution=resolution)
            new_query = new_query.reshape(-1, 1)
            #print(query, new_query)
            bits_agree = np.logical_and(query-epsilon < new_query , 
                                 query+epsilon > new_query).sum()
            bits_changed = n-bits_agree #(new_query!=query).sum()
            query=new_query
            i+=1

            if renorm_query:
                query = query/np.linalg.norm(query)

            if plot and image: 
                plt.imshow(query.reshape(image_len,image_len))
                plt.title('Autoassociative Attention result after: '+str(i)+' iterations')
                plt.show()

        if image: 
            plt.imshow(query.reshape(image_len,image_len))
            plt.title('Autoassociative Attention result after: '+str(i)+' iterations')
            plt.show()

        #print("Attention number of iterations till convergence", i)
        if print_each_result:
            print('Percentage pixels that agree with (noisy) query', np.logical_and(init_query-epsilon < query , init_query+epsilon > query).sum()/n)
            print('epsilon value being used for this equality check =', epsilon)

        converged_queries.append(query)
            
    return converged_queries

def Hopfield(params, raw_flat_patterns, queries_tuple, plot=True, image=False):
    
    queries = queries_tuple[0]
    
    # patterns should be 2D and the columns should be unique patterns. 
    assert len(raw_flat_patterns.shape) ==2, 'patterns are wrong dimenions.'
    
    n, r, m, h_dist, max_iterations, image_len, print_each_result = params
    #make the raw patterns binary:
    P_a_pm = ((raw_flat_patterns>0).astype(int) *2)-1
    
    weights = (P_a_pm@P_a_pm.T)/n
    weights = weights * (1-np.eye(n))
    
    # +-1 the queries:
    converged_queries = []
    for query in queries: 
        init_query = ((query>0).astype(int)*2)-1
        query = np.copy(init_query)
        
        # positive number to trick the while loop
        bits_changed = 100
        i = 0 # iteration tracker
        
        while bits_changed > 0 and i<max_iterations:
        
            new_query = g_hop( weights@query )
            
            #print('query updated by:', (new_query!=query).sum(), 'bits! in iteration:', i)
            bits_changed = (new_query!=query).sum()
            query=new_query
            i+=1
            if plot and image: 
                plt.imshow(query.reshape(image_len,image_len))
                plt.title('OG Hopfield result after: '+str(i)+' iterations')
                plt.show()
              
        if image:
            plt.imshow(query.reshape(image_len,image_len))
            plt.title('OG Hopfield result after: '+str(i)+' iterations')
            plt.show()

        if print_each_result:
            print('Percentage pixels that agree with (noisy) query', (init_query==query).sum()/n)

        converged_queries.append(query)
            
    return converged_queries