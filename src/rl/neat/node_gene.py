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


    def copy(self):
        return NodeGene(self.innovation_nb, self.connections, self.output, self.layer_nb)


    def __eq__(self, node_gene):
        return self.innovation_nb == node_gene.innovation_number
        

    def __str__(self):
        return f'NodeGene[innovation_nb={self.innovation_nb}, connections={self.connections}, \
        output={self.output}, layer_nb={self.layer_nb}]'