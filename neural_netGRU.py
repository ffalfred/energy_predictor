##PACKAGES##
import numpy as np
import theano
import theano.tensor as T
import lasagne
import os
import math
import importlib
from dropout_seq import DropoutSeqPosLayer
##-------##



def neuralnet(post_hidden_1,post_hidden_2,post_hidden_3,pre_hidden_1,pre_hidden_2,n_hid_rec,grad_clip,drop_perm,drop_hid,size_batch,max_length,n_features,input_var=None,mark_var=None):
    l_in = lasagne.layers.InputLayer(shape=(size_batch,max_length,n_features),input_var=input_var)
    l_mask = lasagne.layers.InputLayer(shape=(size_batch,max_length),input_var=mark_var)
    l_drop = DropoutSeqPosLayer(l_in, drop_perm)
    if pre_hidden_1 != 0:
        l_reshape = lasagne.layers.ReshapeLayer(l_drop,((size_batch*max_length,n_features)))
        l_drop = lasagne.layers.DropoutLayer(l_reshape, p=drop_hid)
        l_dense = lasagne.layers.DenseLayer(l_drop, num_units=pre_hidden_1, nonlinearity=lasagne.nonlinearities.tanh)
        l_drop = lasagne.layers.ReshapeLayer(l_dense,((size_batch,max_length,pre_hidden_1)))
        n_features=pre_hidden_1
    if pre_hidden_2 != 0:
        l_reshape = lasagne.layers.ReshapeLayer(l_drop,((size_batch*max_length,n_features)))
        l_drop = lasagne.layers.DropoutLayer(l_reshape,p=drop_hid)
        l_dense = lasagne.layers.DenseLayer(l_drop, num_units=pre_hidden_2, nonlinearity=lasagne.nonlinearities.tanh)
        l_drop = lasagne.layers.ReshapeLayer(l_dense,((size_batch,max_length,pre_hidden_2)))
        n_features=pre_hidden_2
    gate_parameters_hidden = lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),W_cell=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.),nonlinearity=lasagne.nonlinearities.tanh)
    gate_parameters = lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),W_cell=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.))
    l_drop = lasagne.layers.DropoutLayer(l_drop,p=drop_hid)
    l_gru = lasagne.layers.recurrent.GRULayer(
        l_drop, n_hid_rec,
        mask_input=l_mask,
        resetgate=gate_parameters,updategate=gate_parameters,hidden_update=gate_parameters_hidden,
        learn_init=True, grad_clipping=grad_clip)
    l_gru_back = lasagne.layers.recurrent.GRULayer(
        l_drop, n_hid_rec,
        mask_input=l_mask,
        resetgate=gate_parameters,updategate=gate_parameters,hidden_update=gate_parameters_hidden,
        learn_init=True, grad_clipping=grad_clip, backwards=True)
    l_conc = lasagne.layers.ConcatLayer([l_gru, l_gru_back],axis=2)
    l_dense = lasagne.layers.ReshapeLayer(l_conc, (-1, 2*n_hid_rec))
    if post_hidden_1 != 0:
        l_drop = lasagne.layers.DropoutLayer(l_dense,p=drop_hid)
        l_dense = lasagne.layers.DenseLayer(l_drop, num_units=post_hidden_1, nonlinearity=lasagne.nonlinearities.tanh)
    if post_hidden_2 != 0:
        l_drop = lasagne.layers.DropoutLayer(l_dense,p=drop_hid)
        l_dense = lasagne.layers.DenseLayer(l_drop, num_units=post_hidden_2, nonlinearity=lasagne.nonlinearities.tanh)
    if post_hidden_3 != 0:
        l_drop = lasagne.layers.DropoutLayer(l_dense,p=drop_hid)
        l_dense = lasagne.layers.DenseLayer(l_drop, num_units=post_hidden_3, nonlinearity=lasagne.nonlinearities.tanh)
    l_dense = lasagne.layers.DenseLayer(l_dense, num_units=5, nonlinearity=lasagne.nonlinearities.linear)
    l_out = lasagne.layers.ReshapeLayer(l_dense, ((size_batch, max_length, 5)))
    return l_out
