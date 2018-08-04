'''
Graph of operations
'''


class ActiveGraph:
    '''A graph of nodes each of which is working in an independent thread'''
    def __init__(self, nodes=None):
        self.nodes = []
        if nodes is not None:
            self.nodes.extend(nodes)

    def start(self):
        '''Starts all nodes for producing/consuming data'''
        for node in self.nodes:
            node.start()

    def stop(self):
        '''Stops all nodes'''
        for node in self.nodes:
            node.stop()        