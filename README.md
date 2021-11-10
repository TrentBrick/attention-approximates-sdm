# SDM Is All You Need - Codebase

This is all of the code used to run analyses in the paper "Attention Approximates Sparse Distributed Memory" by Trenton Bricken and Cengiz Pehlevan. 

Paper: https://openreview.net/forum?id=WVYzd7GvaOM&noteId=l-hU8Fav3x#all
Blog post: https://www.trentonbricken.com/Attention-Approximates-Sparse-Distributed-Memory/

## Abstract

While Attention has come to be an important mechanism in deep learning, there remains limited intuition for why it works so well. Here, we show that Transformer Attention can be closely related under certain data conditions to Kanerva's Sparse Distributed Memory (SDM), a biologically plausible associative memory model. We confirm that these conditions are satisfied in pre-trained GPT2 Transformer models. We discuss the implications of the Attention-SDM map and provide new computational and biological interpretations of Attention.

## Summary of Paper

The main contribution of this paper is to show that the Sparse Distributed Memory (SDM) theory developed in 1988 for how memories are written to and read from neurons, is a very close approximation to the heuristically developed and powerful Transformer Attention. This connection is compelling because SDM has biologically plausibility with the cerebellum in particular. SDM has a number of additional desireable properties that may lead to improvements in Deep Learning including (citations and explations for these statements provided in the paper): 

* Capable of modelling both auto and heteroassociative relationships.
* Symbolic representations enabling variable binding, learning from example, analogical reasoning, and generalization.
* Sparsity providing computational efficiency and robustness to noise. 
* Biological plausibility with striking similiarities to the cerebellum. Similarities that warrant further investigation are also present in cortical columns, the hippocampus, dorsal cochlear nucleus, and olfactory system in humans, insects and potentially even cephalopods.
* Psychological plausibility including explaining the robust, distributed nature of memories, speed of recognition, tip of the tongue phenomena, Small World network between concepts. 
* Additional strong similarities to the Neural Turing Machine (NTM), and Differentiable Neural Computer (DNC).

## Description of the Codebase

### Jupyter Notebooks:

Used to run all code.

* Softmax_Circle_Approx.ipynb - Computes the approximate circle intersection and shows how it relates to the softmax via the log linear regression to fit Beta in the exponential. This is the core contribution of our paper. 

* Exp_Approx_Circle_Intersect.ipynb - Implements and tests how well the exponential upper and lower bounds analytically derived for the circle intersection perform. 

* SDM_Experiments.ipynb - Calls on functions in Implementations_Associative_Memory.py and Data_Processing_Associative_Memory.py to test all of the Associative Memory algorithms considered: Neuron Based SDM; Pattern Based SDM with Infinite Neurons; Pattern Based SDM with Finite Neurons; Hopfield Network; Binary SDM with Attention with learnt Beta; SDM Attention with learnt Beta; Transformer Attention.

* LearnProjections.ipynb - Also calls on functions in Implementations_Associative_Memory.py to learn a projection matrix for the MNIST and CIFAR datasets before testing how it affects the performance of continuous vectors that use three different weightings: Binary SDM Circle Intersection, Continuous SDM Hypersphere Cap Intersection, Attention Softmax with a Beta fitted to Binary SDM. 

* Neuron_Address_Distribution.ipynb - Computes the probability that at least one neuron is within a given Hamming distance of a random query.

* SDM_Critical_Distances.ipynb - Plots the Critical Distances under different parameter assumptions. 

* HugFace/Transformer_Empirical_Analysis.ipynb - Computes the Betas used in the trained GPT models with the decided upon text inputs. This jupyter notebook is in this directory that implements a customized version of the Hugging Face transformer repo: https://github.com/huggingface/transformers. It was necessary to modify the code base in order to get out the query matrices before their dot product with the keys in the softmax operation. 

* Parse_KeyQ_Norm_Betas.ipynb - Parses and plots the KeyQuery Norm learnt Beta values. 

* Compute_Difference_In_Circle_Intersects.ipynb - Computing how the circle intersection implementations are different from those presented in the SDM book. Also comparing the Circle Intersection equation derived in the Appendix to that of the book. Finally, comparing the associated variance equation from the book with that of Jaeckel's Alterative SDM Design (presented and outlined in the paper Appendix). 

* Optimal_d.ipynb - Computing the Signal to Noise Ratio and Memory Capacity Optimal Hamming Distances.

* Miscellaneous.ipynb - the name says it all. Different experiments and functions not used in the paper. 

#### Python Scripts:

Supporting functions for the Jupyter Notebooks.

* SDM_Circ_Inter_Funcs.py - Contains lots of heavily used functions including implementing the circle intersection function and fitting the log linear regression to the circle intersection.

* Implementations_Associative_Memory.py - Handles the algorithmic implementations of all Associative Memory models considered.

* utils_LearningProjections.py - Called by LearnProjections.ipynb, leverages functions from Implementations_Associative_Memory.py but wraps them in Pytorch backpropagation to learn the projection matrix. 

* Data_Processing_Associative_Memory.py - Applies random perturbations to continuous and binary data inputs to then evaluate the autoassociative convergence properties of various algorithms. 

#### Folders:

* figures/ - contains all of the figures used in the paper and additional ones. Aside from those generated by HugFace/Transformer_Empirical_Analysis.ipynb that are located in the next bullet point:

* HugFace/GPT2Outputs/ - contains all of the GPT2 Transformer analysis figures. Generated by HugFace/Transformer_Empirical_Analysis.ipynb.

* trained_weights/ - trained weights of the projection matrix for each dataset, Hamming radius and random initalization. 

#### Data:

* KeyQuery_Norm_Learnt_Betas.txt - Learnt Beta values from the Trained Transformer models of the paper: A. Henry, Prudhvi Raj Dachapally, S. Pawar, and Yuxuan Chen. Query-key normalization for transformers. In EMNLP, 2020.

* HugFace/text_inputs.txt - line separated text inputs put into GPT2 to infer it's effective Betas. This text is used by HugFace/Transformer_Empirical_Analysis.ipynb.

## Dependencies

Tested with Python 3.7.5 (should work with Python 3.5 and higher).

To run HugFace/Transformer_Empirical_Analysis.ipynb you will need to install Pytorch 1.5.1 (using CUDA or not depending on if you have a GPU)
<https://pytorch.org/get-started/locally/> 

If using Pip out of the box `cd` to this directory then use: 
`pip3 install -r SDM/requirements.txt`

If using Conda then ensure pip is installed with conda and then run the same above code.  

Do not install (or uninstall if it is already installed) HuggingFace/transformers. As you will need to run the customized version implemented in the HugFace/ directory. cd to this directory then run: 
`pip install -e .`
In trying to run this there may be a couple additional random dependencies it expects like tdqm but these are straightforward to install when and if prompted. 

## Acknowledgements:

Thanks to the open source community, friends and advisors for making this research possible. This includes but is not limited to: 

Dr. Gabriel Kreiman, Alex Cuozzo, Miles Turpin, Dr. Pentti Kanerva, Joe Choo-Choy, Dr. Beren Millidge, Jacob Zavatone-Veth, Blake Bordelon, Nathan Rollins, Alan Amin, Max Farrens, David Rein, Sam Eure, Grace Bricken, and Davis Brown for providing invaluable inspiration, discussions and feedback. Special thanks to Miles Turpin for help working with the Transformer model experiments. We would also like to thank the open source software contributors that helped make this research possible, including but not limited to: Numpy, Pandas, Scipy, Matplotlib, PyTorch, HuggingFace, and Anaconda.

## Codebase Author:

* **Trenton Bricken** - [trentbrick](https://trentbrick.github.io/)

## License:

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
