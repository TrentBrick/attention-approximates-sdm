"""
Author: Trenton Bricken @trentbrick

Implements the following Associative Memory Algorithms: 

* Binary_Neuron_SDM - Binary vector space. Biologically Plausible. NB. This algorithm takes a long time to run because of the number of Hamming distance computations that must be done (m patterns * r neurons to initialize memory value vectors), then m operations for every query update. It remains slow and memory intensive even with the parallelized version used here. 

* Binary_SDM - Binary vector space. Assumes there are $r=2^n$ neurons in the space. Uses the circle intersection equation that should approximate a binary version of Attention well. 

* Binary_SDM_NeuronLimit - Same as the above but enforces that there are in fact only $r$ neurons in the space. 

* Binary_SDM_BFit_Attention - Binary vector space. Weights for each vector during query update use the softmax from Attention with a beta value fit to the binary circle intersection. 

* Cont_SDM - Continuous vector space. Uses the continuous version of SDM where it computes the intersection of the caps of cones on a hypersphere. 

* Cont_SDM_NeuronLimit - Same as above but enforces a finite number of neurons.

* Cont_Binary_SDM - Continuous vector space but takes cosine similarities between vectors, maps them to hamming distances, and then uses the binary SDM circle intersection to determine vector weightings. 

* Cont_SDM_BFit_Attention - Continuous vector space. Uses the softmax from Attention with a beta value fit to the binary circle intersection.

* Cont_SDM_CFit_Attention - Same as above but uses a beta value fit to the continuous circle intersection equation.  

All use Pytorch in a batched fashion for rapid GPU acceleration. 
Throughout the script you will see ways that we handle numerical errors that emerge from pytorch floating point operations.

"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import binom
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import ray
import psutil
from SDM_Circ_Inter_Funcs import *
from Data_Processing_Associative_Memory import *
import torch.optim as optim
import torch.nn.functional as F
from functools import partial

def imshow(img, show_now=True):
    """
    Used by Pytorch grid function to display lots of images at the same time using the same visualization range. 
    """
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if show_now: # want this false if trying to save figure elsewhere before showing it. 
        plt.show()

def fit_beta(n,r, hamm_radius, cont_circle_intersect, plot=False, return_bias=False):
    """
    Fit a beta coefficient to a given SDM binary circle intersection defined by its dimensional n, number of neurons r, and the hamming radius hamm_radius. 
    """
    dvs = np.arange(0,hamm_radius).astype(int)
    cs_v = hamm_to_cosine(dvs,n) # for the softmax
    cont_cs_v = np.copy(cs_v)
    cont_cs_v[0] = cont_cs_v[0] - 1e-15 # if the cosine similarity equals 1 this throws an error so need to reduce it by a small epsilon. 
    assert cont_cs_v[0] != 'should not be 1 else will throw a division error in the equation.'

    if cont_circle_intersect:
        res = cap_intersection(n, cont_cs_v, hamm_radius, r)
        out = fit_beta_regression(n, cont_cs_v, res, return_bias=return_bias, ham_input=False )
    else: 
        res = expected_intersection_lune(n, dvs, hamm_radius, r)
        out = fit_beta_regression(n, cs_v, res, return_bias=return_bias, ham_input=False )
    res = np.asarray(res)
    
    if return_bias: 
        beta = out[0]
        bias = out[1]
    else: 
        beta = out

    if plot: 
        plt.plot(dvs, res/res.sum(), label='Truth')
        plt.plot(dvs, softmax(cs_v,beta), label="Softmax w/ learned "+ r"$\beta$")
        plt.legend()
        plt.title("Using continuous circle intersection="+str(cont_circle_intersect))
        print('learned softmax beta vs attention beta:', beta, 1/np.sqrt(n))
        plt.xlabel('Hamming Distance between circles')
        plt.ylabel('Normalized Weight')
        plt.show()
    
    if return_bias: 
        return beta, bias
    return beta

# Algorithms query update rules in binary or continuous space. 
# All assume that matrices have vectors as their columns.

def update_Binary_Neuron_SDM(X_a, X_v, h_v, hamm_radius, queries):
    query_neuron_inter = (torch_hamm_dist(queries, X_a) <=  hamm_radius).type(torch.float)
    return (( ( X_v@query_neuron_inter ) / ( h_v @ query_neuron_inter ) ) >0.5).type(torch.int)

def update_Cont_Attention(P_a, P_p, beta, queries, full_dim_queries=None):
    # ignore full_dim_queries parameter. this is a hacky pass in function that is used due to numerical stability
    # issues in the 'update_using_cache' function. 
    return P_p@F.softmax( (P_a.T@queries)*beta, dim=0 )
    
def update_Binary_Attention(P_a, P_p, beta, queries):
    cos_sims = hamm_to_cosine(torch_hamm_dist(queries, P_a ), queries.shape[0]).type(torch.float)
    return ((P_p.type(torch.float)@F.softmax( cos_sims*beta, dim=0 ) )>0.5).type(torch.int)

def update_using_cache(P_a, P_p, cached_intersects, enforce_num_neurons,
                            intersect_type, device, queries, full_dim_queries = None):
    """
    Pytorch updating of the query using cached circle intersection values. 
    This is called by Binary_SDM, Cont_SDM, Cont_Binary_SDM and their NeuronLimited versions.
    
    When using the continuous circle intersection, it handles values on the log scale to improve
    its numerical stability. This is only when 'FullCont' is the intersect type. 
    
    args::
    'intersect_type' can be: FullBinary, ContUsingBinaryInter, FullCont 
    'enforce_num_neurons': true when algorithm is NeuronLimited
    'full_dim_queries': used for when projections are being learnt to restore queries that 
    update to nan values as perturbations are too large to their original input. 
    
    returns::
    new_queries: updated queries
    """
    n = queries.shape[0]
    if intersect_type=='FullBinary':
        cache_inds = torch_hamm_dist(queries, P_a )
    elif intersect_type=='ContUsingBinaryInter':
        dotps = P_a.T@queries
        # convert to Hamming distances: 
        cache_inds = cosine_to_hamm(dotps, n)
        assert (cache_inds < 0).sum()==0 and (cache_inds>n).sum()==0, 'cache inds out of bounds'
        
    elif intersect_type=='FullCont':
        dotps = P_a.T@queries
        # ensuring that the dotproducts are mapped to the correct cached index value. 
        cache_inds = ( (len(cached_intersects)/2*(dotps + 1))-1 )
    else: 
        raise Exception("Don't know intersect type!")
    intersects = cached_intersects[cache_inds.type(torch.long)]
    if enforce_num_neurons: 
        if intersect_type=='FullCont': # need to use log values
            decimals = torch.exp(intersects)%1
            one_more_neuron_sample = (torch.rand(intersects.shape, device=device)<decimals).type(torch.float)
            # smallest epsilon to avoid log by 0 errors:
            intersects = torch.log(torch.floor(torch.exp(intersects))+one_more_neuron_sample + 1e-30) 
            # need to enforce hard cutoff. 
        else: 
            #print('pre', intersects[:,0])
            decimals = intersects%1
            one_more_neuron_sample = (torch.rand(intersects.shape, device=device)<decimals).type(torch.float)
            #print('intersects pre floor', intersects, cache_inds)
            intersects = torch.floor(intersects)+one_more_neuron_sample
            #print('post', intersects[:,0])
    
    # get pattern weightings
    if intersect_type=='FullCont': 
        # needed as intersects are in log space. 
        new_queries = (P_p@torch.exp(intersects) )/torch.exp(intersects).sum(0)
    else: 
        new_queries = (P_p.type(torch.float)@intersects.type(torch.float))/intersects.sum(0)
    
    #print(intersects, new_queries[:,0])
    
    if intersect_type=='FullBinary': # binarizing. 
        new_queries = (new_queries>0.5).type(torch.int)
        
    # any intersect that is 0 gives nan values for its update. 
    # these need to be found and ignored to avoid errors. 
    # the query should just not update at all.
    nan_from_no_intersects = torch.isnan(new_queries).sum(0) >0
    if nan_from_no_intersects.sum()>0:
        if full_dim_queries is not None: 
            new_queries[:, nan_from_no_intersects] = full_dim_queries[:, nan_from_no_intersects]
        else: 
            new_queries[:, nan_from_no_intersects] = queries[:, nan_from_no_intersects]
        
    # if intersecttions are larger than 1e-20 but still very small the whole query may still be zeros
    # when its L2 norm is computed which throws errors. This is checked here and fixed  by setting it 
    # back to its pre update version. This happens very rarely. Once every ~200K query perturbations 
    # and only at the largest perturbation distances. 
    #print(new_queries.sum(0), new_queries[:,0])
    if intersect_type!='FullBinary':
        all_zs = new_queries.sum(0)==0.0
        if all_zs.sum() >0:
            if full_dim_queries is not None: 
                new_queries[:, all_zs] = full_dim_queries[:, all_zs]
            else: 
                new_queries[:, all_zs] = queries[:, all_zs]
    return new_queries

def torch_run_algo(algo, P, queries, P_a, P_p, cached_intersects, beta, W_k = None):
    """
    All algorithm strings are processed here to set to the relevant query update function. 
    Calls 'convergence_loop' function that iteratively updates the query. 
    
    args::
    algo - string for algorithm name
    P - sdm_parameters
    P_a - pattern addresses
    P_p - pattern pointers. These are the same as the addresses except for when projection matrices
    are learnt where the addresses are in the learnt latent space but the pointers are for the full
    dimensional images. 
    cached_intersects - binary or continuous (in log space)
    beta - fit to binary or continuous circle intersection with relevant sdm parameters
    W_k - if provided used to project the updated query back into the latent space. 
    
    returns::
    fully converged updated batch of queries. dimensions: batch_size x pattern pointer dimension
    """ 
    assert len(P_a.shape) ==2, 'patterns are wrong dimenions.'
    if "Attention" in algo:
        if P.binary_data:
            query_update_func = partial(update_Binary_Attention, P_a, P_p, beta )
        else: 
            query_update_func = partial(update_Cont_Attention, P_a, P_p, beta )
            
    if "NeuronLimit" in algo and P.r is not None: 
        enforce_num_neurons_in_convergence = True
    else: 
        enforce_num_neurons_in_convergence = False
        
    # Specific setups for each function: 
    if algo =="Binary_Neuron_SDM":
        # generate random neuron addresses:
        X_a = torch.Tensor(binom.rvs(1, 0.5, size=(P.n,P.r))).to(P.device)
        X_a_P_a_interactions = torch_hamm_dist(X_a, P_a) <=  P.hamm_radius # converting back to columns
        h_v = X_a_P_a_interactions.sum(0).type(torch.float)
        X_v = P_p.type(torch.float)@X_a_P_a_interactions.type(torch.float)
        query_update_func = partial(update_Binary_Neuron_SDM, X_a, X_v, h_v, P.hamm_radius)
       
    elif algo =="Binary_SDM" or algo =="Binary_SDM_NeuronLimit":
        query_update_func = partial(update_using_cache, P_a, P_p, cached_intersects, 
                                    enforce_num_neurons_in_convergence, "FullBinary", P.device)
        
    elif algo =="Cont_SDM" or algo =="Cont_SDM_NeuronLimit":
        query_update_func = partial(update_using_cache, P_a, P_p, cached_intersects, 
                                    enforce_num_neurons_in_convergence, "FullCont", P.device)
        
    elif algo =="Cont_Binary_SDM" or algo =="Cont_Binary_SDM_NeuronLimit":
        query_update_func = partial(update_using_cache, P_a, P_p, cached_intersects, 
                                    enforce_num_neurons_in_convergence, "ContUsingBinaryInter", P.device)
        
    # All are already established by the Attention checker above.   
    elif algo =="Binary_SDM_BFit_Attention":
        pass
    elif algo =="Cont_SDM_BFit_Attention":
        pass 
    elif algo =="Cont_SDM_CFit_Attention":
        pass
    elif algo =="Hopfield":
        raise Exception("Not implemented here.")
        
    else:
        raise Exception("Don't recognize the name of the function")
    
    return convergence_loop(query_update_func, P, queries, W_k )

def torch_cosine_sims(A, B):
    # assumes that vectors are columns. Returns row of cosine similarities in numpy
    return (A/torch.norm(A, dim=0,keepdim=True)*B/torch.norm(B, dim=0,keepdim=True)).sum(0).cpu().detach().numpy()

def check_caches_work_for_num_neurons(r, hamm_radius, use_binary_intersect, b_cached_inters, log_c_cached_inters):
    """
    Depending upon the size of the hamming radius, there can be too few neurons in the space for convergence to occur. 
    This is checked here for the binary and (log) continuous circle intersections depending on which algorithm is being 
    used. 
    
    """
    if r is not None and (use_binary_intersect and torch.round(b_cached_inters[0])==0.0):
        print("r of", r, "is too small for d radii of", hamm_radius, "largest intersections",
              b_cached_inters[:5])
        return 'continue'

    if r is not None and (not use_binary_intersect and torch.round(torch.exp(log_c_cached_inters[-1]))==0.0):
        print("r of", r, "is too small for d radii of", hamm_radius, "largest intersections", torch.exp(log_c_cached_inters[-5:]))
        return 'continue'
    return None

def convergence_loop(query_update_func, P, queries, W_k=None):
    """
    Ensures that each query in the batch has converged or hits the 
    max number of convergence steps allowed. 
    
    args::
    query_update_func - partial function that has all parameters specific to each algorithm already set. 
    It just needs the query as input. 
    
    P - sdm_params
    queries - vectors in the matrix. 
    W_k - if given used to project the queries in their latent space. 
    """
    
    bits_changed_sum = 100000 # some large number
    i = 0
    # will keep going as long as none of them are satisfied. 
    while bits_changed_sum > 0 and i<P.max_conv_steps: 
        
        '''plt.imshow(queries[:,0].cpu().numpy().reshape(28,28))
        plt.show()'''
        
        if W_k is not None: 
            pQs = W_k@queries
            # L2 norm all
            pQs = pQs/torch.norm(pQs, dim=0, keepdim=True)
            new_queries = query_update_func(pQs, full_dim_queries=queries)
        else: 
            new_queries = query_update_func(queries)
        if P.binary_data:
            bits_changed_sum = (new_queries!=queries).sum()
        else: 
            # (P.n_base*P.batch_size)-
            bits_changed_sum = torch.logical_or(queries-P.epsilon > new_queries , 
                             queries+P.epsilon < new_queries).sum()
        queries = new_queries
        if not P.binary_data and P.renorm_cont_queries:
            # renorming the query for all Attention operations here.
            # dont want to do this have learnt a projection matrix as 
            # returned query here is in image space not L2 norm latent space. 
            queries = queries/torch.norm(queries, dim=0, keepdim=True)
        i+=1
    return queries

'''

CODE FOR THE HOPFIELD NETWORK THAT WE NO LONGER RUN COMPARISONS FOR (BEYOND THE SCOPE OF THIS WORK.)
NOT PYTORCH OR BATCH OPTIMIZED. 

def g_hop(X): # majority update for bipolar values
    return ((X>0).astype(int)*2)-1

def Hopfield(params, raw_flat_patterns, queries_tuple, plot=True, is_image=False):
    
    queries = queries_tuple[0]
    
    # patterns should be 2D and the columns should be unique patterns. 
    assert len(raw_flat_patterns.shape) ==2, 'patterns are wrong dimenions.'
    
    n, r, m, hamm_radius, max_iterations, image_len, print_each_result = params
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
            if plot and is_image: 
                plt.imshow(query.reshape(image_len,image_len))
                plt.title('OG Hopfield result after: '+str(i)+' iterations')
                plt.show()
              
        if is_image:
            plt.imshow(query.reshape(image_len,image_len))
            plt.title('OG Hopfield result after: '+str(i)+' iterations')
            plt.show()

        if print_each_result:
            print('Percentage pixels that agree with (noisy) query', (init_query==query).sum()/n)

        converged_queries.append(query)
            
    return converged_queries
'''