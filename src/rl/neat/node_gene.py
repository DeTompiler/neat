from src.rl.neat.gene import Gene
from collections import deque


class NodeGene(Gene):
    ''' NodeGene inherits from Gene class
        - layer_nb is used for forward propagation
        - index_in_layer may need to be used for visualization only
    '''

    def __init__(self, innovation_nb, connections=deque(), output=0, layer_nb=0):
        super().__init__(innovation_nb)
        self.connections = connections
        self.output = output
        self.layer_nb = layer_nb


    def copy(self, copy_connections=False):
        connections = self.connections if copy_connections else deque()

        return NodeGene(self.innovation_nb, connections, self.output, self.layer_nb)


    def has_connection_to(self, node):
        for connection in self.connections:
            if connection.node_out == node:
                return True

        return False


    def __eq__(self, node_gene):
        return self.innovation_nb == node_gene.innovation_number
        

    def __str__(self):
        return (f'NodeGene[innovation_nb={self.innovation_nb}, '
        f'connections={[connection.node_out.innovation_nb for connection in self.connections]}, '
        f'output={self.output}, layer_nb={self.layer_nb}]')