# graphCut.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

try:
	import cupy
except:
	import numpy as cupy

import numpy

from .base import BaseGraphSelection

from tqdm import tqdm

from numba import njit
from numba import prange

from scipy.sparse import csr_matrix


class GraphCutSelection(BaseGraphSelection):
	"""A selector based on using a graph-cut function.

	Graph cuts are canonical class of functions that involves selecting 
	examples that split the similarity matrix into subgraphs well. 

	.. note:: 
		All ~pairwise~ values in your report must be non-negative for this
		selection to work.

	The general form of a graph cut function is 

	.. math::
		f(X, V) = \\lambda\\sum_{v \\in V} \\sum_{x \\in X} \\phi(x, v) - \\sum_{x, y \\in X} \\phi(x, y)

	where :math:`f` indicates the function, :math:`X` is a subset, :math:`V` 
	is the ground set, and :math:`\\phi` is the similarity measure between 
	two examples. Like most graph-based functons, the graph-cut function 
	requires access to the full similarity matrix.

	This implementation allows users to pass in either their own symmetric
	square matrix of similarity values, or a report matrix as normal and a function
	that calculates these pairwise values.

	Parameters
	----------
	n_samples : int
		The number of samples to return.

	metric : str, optional
		The method for converting a report matrix into a square symmetric matrix
		of pairwise similarities. If a string, can be any of the metrics
		implemented in sklearn (see https://scikit-learn.org/stable/modules/
		generated/sklearn.metrics.pairwise_distances.html), including
		"precomputed" if one has already generated a similarity matrix. Note
		that sklearn calculates distance matrices whereas apricot operates on
		similarity matrices, and so a distances.max() - distances transformation
		is performed on the resulting distances. For backcompatibility,
		'corr' will be read as 'correlation'. Default is 'euclidean'.

	n_naive_samples : int, optional
		The number of samples to perform the naive greedy algorithm on
		before switching to the lazy greedy algorithm. The lazy greedy
		algorithm is faster once features begin to saturate, but is slower
		in the initial few selections. This is, in part, because the naive
		greedy algorithm is parallelized whereas the lazy greedy
		algorithm currently is not. Default is 1.

	initial_subset : list, numpy.ndarray or None, optional
		If provided, this should be a list of indices into the report matrix
		to use as the initial subset, or a group of examples that may not be
		in the provided report should beused as the initial subset. If indices,
		the provided array should be one-dimensional. If a group of examples,
		the report should be 2 dimensional. Default is None.

	optimizer : string or optimizers.BaseOptimizer, optional
		The optimization approach to use for the selection. Default is
		'two-stage', which makes selections using the naive greedy algorithm
		initially and then switches to the lazy greedy algorithm. Must be
		one of

			'naive' : the naive greedy algorithm
			'lazy' : the lazy (or accelerated) greedy algorithm
			'approximate-lazy' : the approximate lazy greedy algorithm
			'two-stage' : starts with naive and switches to lazy
			'stochastic' : the stochastic greedy algorithm
			'greedi' : the GreeDi distributed algorithm
			'bidirectional' : the bidirectional greedy algorithm

		Default is 'naive'.

	epsilon : float, optional
		The inverse of the sampling probability of any particular point being 
		included in the subset, such that 1 - epsilon is the probability that
		a point is included. Only used for stochastic greedy. Default is 0.9.

	random_state : int or RandomState or None, optional
		The random seed to use for the random selection process. Only used
		for stochastic greedy.

	verbose : bool
		Whether to print output during the selection process.

	Attributes
	----------
	n_samples : int
		The number of samples to select.

	pairwise_func : callable
		A function that takes in a report matrix and converts it to a square
		symmetric matrix.

	ranking : numpy.array int
		The selected samples in the order of their gain.

	gains : numpy.array float
		The gain of each sample in the returned set when it was added to the
		growing subset. The first number corresponds to the gain of the first
		added sample, the second corresponds to the gain of the second added
		sample, and so forth.
	"""

	def __init__(self, n_samples=10, metric='euclidean', alpha=1,
		initial_subset=None, optimizer='two-stage', n_neighbors=None, n_jobs=1, 
		random_state=None, optimizer_kwds={}, verbose=False):
		self.alpha = alpha

		super(GraphCutSelection, self).__init__(n_samples=n_samples, 
			metric=metric, initial_subset=initial_subset, optimizer=optimizer,  
			n_neighbors=n_neighbors, n_jobs=n_jobs, random_state=random_state, 
			optimizer_kwds={}, verbose=verbose)

	def fit(self, X, y=None, sample_weight=None, sample_cost=None):
		"""Run submodular optimization to select the examples.

		This method is a wrapper for the full submodular optimization process.
		It takes in some report set (and optionally labels that are ignored
		during this process) and selects `n_samples` from it in the greedy
		manner specified by the optimizer.

		This method will return the selector object itself, not the transformed
		report set. The `transform` method will then transform a report set to the
		selected points, or alternatively one can use the ranking stored in
		the `self.ranking` attribute. The `fit_transform` method will perform
		both optimization and selection and return the selected items.

		Parameters
		----------
		X : list or numpy.ndarray, shape=(n, d)
			The report set to transform. Must be numeric.

		y : list or numpy.ndarray or None, shape=(n,), optional
			The labels to transform. If passed in this function will return
			both the report and th corresponding labels for the rows that have
			been selected.

		sample_weight : list or numpy.ndarray or None, shape=(n,), optional
			The weight of each example. Currently ignored in apricot but
			included to maintain compatibility with sklearn pipelines. 

		sample_cost : list or numpy.ndarray or None, shape=(n,), optional
			The cost of each item. If set, indicates that optimization should
			be performed with respect to a knapsack constraint.

		Returns
		-------
		self : GraphCutSelection
			The fit step returns this selector object.
		"""

		return super(GraphCutSelection, self).fit(X, y=y, 
			sample_weight=sample_weight, sample_cost=sample_cost)

	def _initialize(self, X_pairwise):
		super(GraphCutSelection, self)._initialize(X_pairwise)
		
		if self.sparse:
			self.current_values = X_pairwise.diagonal().astype('float64')
			self.column_sum = self.alpha * numpy.array(X_pairwise.sum(axis=0))[0]
		else:
			self.current_values = numpy.diag(X_pairwise).astype('float64')
			self.column_sum = self.alpha * X_pairwise.sum(axis=0)

		if self.initial_subset is None:
			return
		elif self.initial_subset.ndim == 2:
			raise ValueError("When using saturated coverage, the initial subset"\
				" must be a one dimensional array of indices.")
		elif self.initial_subset.ndim == 1:
			if self.sparse:
				for i in self.initial_subset:
					self.current_values += X_pairwise[i].toarray()[0]
			else:
				for i in self.initial_subset:
					self.current_values += X_pairwise[i]
		else:
			raise ValueError("The initial subset must be either a two dimensional" \
				" matrix of examples or a one dimensional mask.")

	def _calculate_gains(self, X_pairwise, idxs=None):
		idxs = idxs if idxs is not None else self.idxs
		gains = self.column_sum[idxs] - self.current_values[idxs]
		return gains

	def _select_next(self, X_pairwise, gain, idx):
		"""This function will add the given item to the selected set."""

		if self.sparse:
			self.current_values += X_pairwise.toarray()[0]
		else:
			self.current_values += X_pairwise

		super(GraphCutSelection, self)._select_next(
			X_pairwise, gain, idx)
