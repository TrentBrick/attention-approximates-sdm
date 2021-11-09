"""
Author: Trenton Bricken @trentbrick

Generates either the MNIST or Random Pattern datasets. 

Selects n_repeats number of queries and randomly perturbs them by the desired amount.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import binom
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import ray
import psutil
from SDM_Circ_Inter_Funcs import space_frac_to_hamm_radius, hamm_to_cosine, cosine_to_hamm, torch_hamm_dist
import torch
import torchvision
import torchvision.transforms as transforms

def cosine_perturb_queries(queries, patterns, P):
    # ensuring it is already L2 normalized. 
    norm_patterns = patterns/torch.norm(patterns, dim=0, keepdim=True)
    assert queries.shape[0] == norm_patterns.shape[0], "Need to transpose the patterns? "
    # Uses Pytorch to take a pattern and provide a desired perturbation to it.
    # Code taken from:
    # https://stackoverflow.com/questions/52916699/create-random-vector-given-cosine-similarity
    init_query_norms = torch.norm(queries, dim=0)
    all_closest = False
    num_attempts = 0
    while not all_closest:
        if P.fix_perturb:
            cs_perturbs = torch.ones((P.batch_size), device=P.device) * hamm_to_cosine( P.perturb , P.n_base)
        else: 
            rand_perturbs = torch.rand((P.batch_size), device=P.device)*(P.max_train_perturb-P.min_train_perturb)+P.min_train_perturb
            cs_perturbs = torch.ones((P.batch_size), device=P.device) * hamm_to_cosine( rand_perturbs , P.n_base)
        us = queries/torch.norm(queries, dim=0, keepdim=True)
        rand_vects = torch.rand(queries.shape, device=P.device)*2 - 1
        target_perps = rand_vects - (rand_vects*us).sum(dim=0)*us
        target_perps = target_perps / torch.norm(target_perps, dim=0, keepdim=True)
        pert_queries = cs_perturbs*us + torch.sqrt(1-cs_perturbs**2)*target_perps
        # resetting to their original norms:
        pert_queries = pert_queries*init_query_norms.unsqueeze(0)
    
        # check that target is still the closest: 
        vs,_ = torch.max(norm_patterns.T@(pert_queries/torch.norm(pert_queries, dim=0,keepdim=True)), dim=0) 
        isclosests = vs <= cs_perturbs+0.00001 # to deal with numerical error.
        if P.ensure_no_closer_images and isclosests.sum() != P.batch_size:
            print("At least one perturbed continuous query is no longer the closest. Re-generating dataset")
            all_closest = False
            if num_attempts>10:
                print("num is closest and perturbation amount:", isclosests.sum(), P.perturb)
                raise Exception("Cant find perturbations that keep as closest")
        else: 
            all_closest = True
        num_attempts+=1
    return pert_queries, isclosests

def hamming_perturb_queries(queries, patterns, P):
    #converting to bipolar to more easily flip 1s and 0s.
    queries = queries*2 - 1
    all_closest = False
    num_attempts = 0
    while not all_closest:
        swap_inds = torch.Tensor(np.asarray([np.random.choice(P.n_base,P.perturb, replace=False) for i in range( P.batch_size )]) ).T.to(P.device).type(torch.long)
        valstoscatter = torch.gather(queries, 0, swap_inds ) *-1
        pert_queries = torch.scatter(queries, 0, swap_inds, valstoscatter)
        pert_queries = (pert_queries+1) //2 # back to binary
        vs,_ = torch.min(torch_hamm_dist(pert_queries, patterns), dim=0) 
        isclosests = vs.type(torch.int) == P.perturb
        #print(torch_hamm_dist(queries, patterns)[:,0], (queries[:,0]==patterns[:,0]).sum(),isclosests.sum(), vs.type(torch.int) , P.perturb)
        if P.ensure_no_closer_images and isclosests.sum() != P.batch_size:
            print("At least one perturbed binary query is no longer the closest. Re-generating dataset")
            all_closest = False
            if num_attempts>10:
                raise Exception("Cant find perturbations that keep as closest")
        else: 
            all_closest = True
        num_attempts+=1
    return pert_queries, isclosests
    
def perturb_queries(b_queries, b_patterns, c_queries, c_patterns, P):
    cos_perts, cos_isclosests = cosine_perturb_queries(c_queries, c_patterns,P) 
    bin_perts, bin_isclosests = hamming_perturb_queries(b_queries, b_patterns, P)
    return bin_perts, cos_perts
