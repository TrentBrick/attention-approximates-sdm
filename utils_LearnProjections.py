"""
Author: Trenton Bricken @trentbrick

Implements functions for learning the projection matrix (and beta if desired). 
Calls functions in "Implementations_Associative_Memory.py" but wraps them in 
gradient descent dynamics and actual use of the projection into a smaller latent space. 

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
from Implementations_Associative_Memory import *
import torch.optim as optim
import torch.nn.functional as F
from functools import partial

def train_projection(W_k, P, trainloader, keys, beta):
    
    ''' 
    Optimization to learn the key projection matrix.

    args:: 
    W_k = Key projection matrix
    P = training parameters
    trainloader = iterates through data batches
    keys = all of the patterns
    beta = can be learnt or provided to approximate the SDM circle intersection

    returns:: 
    W_k = Key projection matrix Torch.Tensor
    if learning beta then also returns this parameter. 
    '''
    if P.learn_beta: 
        optimizer = optim.Adam([W_k, beta], lr=P.lr)
    else: 
        optimizer = optim.Adam([W_k], lr=P.lr)
    losses = []
    
    for e in range(P.proj_epochs): # to train the projection matrix
        cosine_sims = []
        for batch_ind, queries in enumerate(trainloader): 
            # get a minibatch
            P.batch_size = len(queries)
            queries = queries.to(P.device).T # making the queries into columns. 
            targets = torch.clone(queries)
            
            # training loop:
            optimizer.zero_grad()   # zero the gradient buffers
            
            # perturb the queries first: 
            queries, isclosests = cosine_perturb_queries(queries, keys, P)
            
            #project keys
            pKs = W_k@keys
            pKs = pKs/torch.norm(pKs, dim=0,keepdim=True)

            # run convergence on the queries
            query_update_func = partial(update_Cont_Attention, pKs, keys, beta )
            if P.train_multiple_steps:
                queries = convergence_loop(query_update_func, P, queries, W_k=W_k)
            else: 
                pQs = W_k@queries
                # L2 norm all
                pQs = pQs/torch.norm(pQs, dim=0, keepdim=True)
                queries = query_update_func(pQs)
            # reconstruction loss after convergence iterations. 
            loss = ((queries - targets)**2).sum()
            loss /= P.batch_size
            loss.backward()
            optimizer.step() 

            losses.append(loss.item())
            cosine_sim = (queries/torch.norm(queries, dim=0,keepdim=True)*targets/torch.norm(targets, dim=0,keepdim=True)).sum(0).cpu().detach().numpy()
            cosine_sims += list(cosine_sim)
            
            if P.plot_batches: 
                if P.batch_ind%P.plot_batches==0:
                    plt.plot(losses)
                    plt.show()
                    plt.imshow(W_k[:100,:100].cpu().detach().numpy())
                    plt.show()
                    print( "All -- Average cosine sims", sum(cosine_sims)/len(cosine_sims) )
                    
            

        print("epoch", e)
        if P.learn_beta: 
            print(beta)
        if P.plot_epochs: 
            plt.plot(losses)
            plt.show()
            print( "All -- Average cosine sims", sum(cosine_sims)/len(cosine_sims) )
            print('=======')
        
    # do a final plot of this training run. 
    print("epoch", e) 
    plt.plot(losses)
    plt.show()
    print( "All -- Average cosine sims", sum(cosine_sims)/len(cosine_sims) )
    print('=======')  
    if P.learn_beta: 
        return (W_k, beta)
    else:
        return W_k

def test_projection(algo, perturb_hamms, W_k, P, trainloader, 
                     testloader, keys,
                     tkeys, cached_intersects, beta, plot_results=True):
    """
    Testing the trained projection matrix on all train and test data 
    using all three algorithms: 
    Binary, Continuous and Softmax. 
    
    args::
    perturb_hamms = list of perturbation Hamming distances 
    W_k = trained projection matrix
    P = testing parameters
    trainloader = Pytorch Dataloader for training data. 
    testloader = testing data.
    keys = train data keys
    tkeys = test keys
    cached_intersect = that shouljd be used by this algorithm 
    
    returns::
    cos_res = cosine similarity of converged query to its target pattern
    """
    cos_res = {'train':[], 'test':[]}
    for perturb in perturb_hamms:
        P.perturb=perturb
        if plot_results:
            print("!!! Perturb is:", perturb)
        with torch.no_grad(): 
            for train_o_test, loader in zip(['train','test'],[trainloader, testloader]):
                if plot_results:
                    print('============',train_o_test,'===========')
                cosine_sims = []
                keys_to_use = keys if train_o_test=='train' else tkeys
                # project keys
                pKs = W_k@keys_to_use
                pKs = pKs/torch.norm(pKs, dim=0,keepdim=True)

                for lind, queries in enumerate(loader):
                    # get a minibatch
                    P.batch_size = len(queries)
                    queries = queries.to(P.device).T # making the queries into columns. 
                    targets = torch.clone(queries)
                    
                    # perturb queries:
                    queries, isclosests = cosine_perturb_queries(queries, keys_to_use, P)
                    
                    queries = torch_run_algo(algo, P, queries, pKs, keys_to_use,
                                                                 cached_intersects, beta, W_k=W_k)
                    cosine_sim = torch_cosine_sims(queries, targets)
                    cosine_sims += list(cosine_sim)
                cosine_sims = np.asarray(cosine_sims)
                cos_res[train_o_test].append(cosine_sims)#sum(cosine_sims)/len(cosine_sims))
                if plot_results:
                    plt.hist(cosine_sims, bins=100)
                    plt.title("Cosine Sims")
                    plt.show()
                    print( "All -- Average cosine sim", sum(cosine_sims)/len(cosine_sims) )
    return cos_res