from neat.gene import Gene
from neat.utils import Math
import numpy as np


class ConnectionGene(Gene):
    ''' ConnectionGene inherits from Gene class
    '''

    def __init__(self, node_in, node_out, weight=None, enabled=None, weight_randomization_factor=1.):
        super().__init__(Math.cantor_pairing(node_in.innovation_nb, node_out.innovation_nb))
        self.node_in = node_in
        self.node_out = node_out
        self.weight = weight if weight is not None else self.get_random_weight(weight_randomization_factor)
        self.enabled = enabled


    def copy(self, node_in=None, node_out=None):
        node_in = self.node_in.copy() if node_in is None else node_in
        node_out = self.node_out.copy() if node_out is None else node_out
        
        return ConnectionGene(node_in, node_out, self.weight, self.enabled)


    def get_random_weight(self, weight_randomization_factor=1.):
        return np.random.rand() * weight_randomization_factor


    def random_weight(self, weight_randomization_factor=1.):
        self.weight = self.get_random_weight(weight_randomization_factor)
    

    def shift_weight(self, weight_shift_factor=1.):
        self.weight = np.random.uniform(-1, 1) * weight_shift_factor

    
    def update_innovation_nb(self):
        self.innovation_nb = Math.cantor_pairing(self.node_in.innovation_nb, self.node_out.innovation_nb)

    
    def set_node_in(self, node_in):
        self.node_in = node_in
        self.update_innovation_nb()
    

    def set_node_out(self, node_out):
        self.node_out = node_out
        self.update_innovation_nb()

    
    def __eq__(self, connection_gene):
        return self.innovation_nb == connection_gene.innovation_nb


    def __str__(self):
        return (f'ConnectionGene[innovation_nb={self.innovation_nb}, node_in={self.node_in.innovation_nb}, '
        f'node_out={self.node_out.innovation_nb}, weight={self.weight}, enabled={self.enabled}]')