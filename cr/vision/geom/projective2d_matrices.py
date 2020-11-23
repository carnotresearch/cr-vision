'''
A set of classes to represent projective transformations
'''

import numpy as np


class Identity:
    '''
    An identity transform definition
    '''
    def __init__(self):
        pass

    def matrix(self):
        '''
        Returns the transformation matrix for identity transformation
        '''
        return np.eye(3)

class Transform:
    '''
    A projective transformation matrix definition
    '''

    def __init__(self, previous):
        self.previous = previous

    def matrix(self):
        '''
        Returns the transformation matrix
        '''
        raise NotImplementedError


class Translation(Transform):
    '''
    Translation
    '''

    def __init__(self, previous, t_x, t_y):
        super().__init__(previous)
        self.t_x = t_x
        self.t_y = t_y


    def matrix(self):
        result = self.previous.matrix()
        result[0, 2] += self.t_x
        result[1, 2] += self.t_y
        return result
