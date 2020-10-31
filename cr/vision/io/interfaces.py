'''
Interfaces for IO
'''


class FrameSequence:
    '''Frame sequence interface'''

    def __next__(self):
        '''Returns next frame (along with its frame number)'''
        raise NotImplementedError

    def __iter__(self):
        return self

    def is_open(self):
        '''Returns if the sequence is open for reading more frames'''
        raise NotImplementedError

    def is_done(self):
        '''Returns if the sequence is over'''
        raise NotImplementedError

    def stop(self):
        '''Stop serving more frames'''
        raise NotImplementedError
