import numpy as np
import theano
import theano.tensor as T
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan
from lasagne.layers import MergeLayer
from lasagne.layers.base import Layer
from lasagne.layers import helper
from lasagne.layers import MergeLayer, Layer, InputLayer, DenseLayer, helper, Gate
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.random import get_rng
import random

class DropoutSeqPosLayer(Layer):
	"""Dropout layer
	Sets all values to zero in a position in the sequence with probability p. See notes for disabling dropout
	during testing. 
	"""
	def __init__(self, incoming, p=0.5, **kwargs):
		super(DropoutSeqPosLayer, self).__init__(incoming, **kwargs)
		self._srng = RandomStreams(get_rng().randint(1, 2147462579))
		self.p = p

	def get_output_for(self, input, deterministic=False, **kwargs):
		if deterministic or self.p == 0:
				return input
		else:
				retain_prob = 1 - self.p
				input_shape = input.shape #this should be (bs,seqlen,nf)
				mask = self._srng.binomial(input_shape[:2], p=retain_prob, dtype=input.dtype)
				return input * mask.dimshuffle(0,1,'x')

