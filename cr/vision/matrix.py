'''
Helper functions to work with matrices
'''
import numpy as np


def random_unique_permutations(num_permutations, permutation_size):
    '''Returns a set of randomly chosen unique permutations

    Example: 

        array([[3, 1, 2, 0],
            [1, 2, 0, 3],
            [2, 0, 3, 1]])

    References:
        * https://stackoverflow.com/questions/45437988/numpy-random-choice-to-produce-a-2d-array-with-all-unique-values/45438143#45438143
    '''
    return np.random.rand(num_permutations, permutation_size).argsort(axis=-1)
