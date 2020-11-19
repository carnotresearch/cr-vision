"""
Utility functions for numpy arrays
"""


def array_to_tuple(x):
    """Converts and array into a tuple"""
    return tuple(x.reshape(1,-1)[0])