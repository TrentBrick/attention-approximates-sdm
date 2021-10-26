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
from SDM_Circ_Inter_Funcs import space_frac_to_hamm_dist
import torch
import torchvision
import torchvision.transforms as transforms

def generate_dataset(pattern_dataset, n, m, max_perturb, cfracs, n_repeats, n_perturbs=6):
    
    target_patterns, ham_dist_queries, cosine_dist_queries = [], [], []
    cosine_perturb_amounts = []
    
    # hamming write distance to use
    
    h_dists = space_frac_to_hamm_dist(n,cfracs)

    if pattern_dataset == "MNIST": 
        image = True
        image_len = 28
        
        transform = transforms.Compose(
            [transforms.ToTensor(), 
             #transforms.Normalize((0.5,), (0.5,))
            ])
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=m,
                                                      shuffle=True, num_workers=0)

        dataiter = iter(trainloader)
        raw_patterns_flat, y_labels = dataiter.next()
        raw_patterns_flat, y_labels = raw_patterns_flat.numpy(), y_labels.numpy()
        print('mnist shape', raw_patterns_flat.shape)
        raw_patterns_flat = raw_patterns_flat.reshape(raw_patterns_flat.shape[0], -1).T
        n = raw_patterns_flat.shape[0]
        if n_perturbs==1:
            perturb_hamms = [max_perturb]
        else: 
            perturb_hamms = list(np.unique(np.linspace(0,max_perturb,n_perturbs).astype(int)))
        # how much to perturb by in terms of hamming distance. 
        #print(perturb_hamms)
        check_if_target_still_closest = False
        # used to keep track of the distances and plot how correlated the space is
        target_dist = []
        closest_dist = []
        same_pattern_class = []
    
    elif pattern_dataset == "RandUnif":
        image=False 
        image_len = None
        # random uniform dist patterns. 

        perturb_hamms = list(set(np.linspace(0,max_perturb,n_perturbs).astype(int)))
        perturb_hamms = np.sort(perturb_hamms)
        check_if_target_still_closest = True 
        raw_patterns_flat = np.random.uniform(-1,1,(n,m))
        
        target_dist = None
        closest_dist = None
        same_pattern_class = None
    # vector norm all of the patterns. 
    raw_patterns_flat = raw_patterns_flat/ np.linalg.norm(raw_patterns_flat, axis=0)
    
    print('perturbations are:', perturb_hamms)
    # doing binary perturbations here
    for perturb in perturb_hamms:
        for i in range(n_repeats):

            # pick a pattern, perturb it by a given hamming distance from this pattern. 
            # ensure it is still the best match. 
            # see if it converges. 
            # may need to increase the number of iterations!
            target_pattern_ind = np.random.choice(m,1)[0]
            target_pattern = raw_patterns_flat[:, target_pattern_ind ].reshape(-1,1)
            
            ####### Hamming distance perturbation: 
            # flip the sign on these
            hd_query = np.copy(target_pattern)

            if pattern_dataset == 'RandUnif':
                hd_query[np.random.choice(n,perturb, replace=False)]*=-1
            elif pattern_dataset == 'MNIST':
                rand_inds = np.random.choice(n,perturb, replace=False)
                inds_set_to_zero = rand_inds[(hd_query[rand_inds]>0).flatten()]
                inds_set_to_max = rand_inds[(hd_query[rand_inds]==0).flatten()]
                hd_query[ inds_set_to_zero ] = 0
                hd_query[ inds_set_to_max ] = 254
                #query[np.random.choice(n,perturb, replace=False)] = np.random.uniform(0,254,perturb).astype('uint8').reshape(-1,1)

            # ensuring that the query is still closest to the target pattern in similarity and hamming distance. 
            if check_if_target_still_closest:
                assert np.argmax( raw_patterns_flat.T@ hd_query ) == target_pattern_ind, 'original pattern is no longer the best match'
                assert np.argmin( cdist( (raw_patterns_flat>0).astype(bool).T, 
                                        (hd_query>0).astype(int).T, metric='hamming')*n  ) == target_pattern_ind, \
                                            'hamming original pattern is no longer the best match'
            else: 
                pattern_dists = cdist( (raw_patterns_flat>0).astype(bool).T, 
                                        (hd_query>0).astype(int).T, metric='hamming')*n
                closest_pattern_ind = np.argmin( pattern_dists  )
                closest_distance = np.min(pattern_dists)
                
                '''print('Closest pattern is the original', closest_pattern_ind==target_pattern_ind)
                print('closest and target inds', closest_pattern_ind,target_pattern_ind)
                print('hamming distance from query to closest pattern', closest_distance)
                print('perturb is:', perturb)'''
                
                target_dist.append(perturb)
                closest_dist.append(closest_distance)
                same_pattern_class.append( y_labels[closest_pattern_ind]==y_labels[target_pattern_ind]  )
                
            
            ham_dist_queries.append( hd_query )
            
            ######## Cosine distance perturbation:
            cs_perturb = 1-((2*perturb)/n)
            
            # https://stackoverflow.com/questions/52916699/create-random-vector-given-cosine-similarity
            rand_vect = np.random.uniform(-1,1,(n,1))
            target_perp = rand_vect - (rand_vect.T@target_pattern)*target_pattern
            target_perp = target_perp / np.linalg.norm(target_perp, axis=0)
            cs_query = cs_perturb*target_pattern + np.sqrt(1-cs_perturb**2)*target_perp
            
            cosine_dist_queries.append(cs_query)
            cosine_perturb_amounts.append(cs_perturb)
            target_patterns.append( target_pattern )

    # for this n the hamming distance is wrong. 
    print('hdists are:', h_dists)
    for h_dist in h_dists: 
        assert h_dist < n/2, 'need to change h dist.'
        
    ham_dist_queries = np.asarray(ham_dist_queries)
    cosine_dist_queries = np.asarray(cosine_dist_queries)
    return raw_patterns_flat, target_patterns, ham_dist_queries, cosine_dist_queries, perturb_hamms, h_dists, image, image_len, (target_dist, closest_dist, same_pattern_class) 