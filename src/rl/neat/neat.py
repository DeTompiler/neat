from collections import deque
from src.rl.neat.connection_gene import ConnectionGene
from src.rl.neat.genome import Genome
from src.rl.neat.node_gene import NodeGene


class Neat:
    ''' Neat class
        - input_size - the input size for each genome
        - output_size - the output size for each genome
        - node_innovation_nb - last used innovation number for a new node
        - nodes - dictionary mapping connection_innovation_nb -> NodeGene
    '''


    def __init__(self, input_size, output_size, **kwargs):
        self.input_size = input_size
        self.output_size = output_size

        self.toggle_probability = kwargs.get('toggle_probability', 0.01)
        self.shift_probability = kwargs.get('shift_probability', 0.01)
        self.random_probability = kwargs.get('random_probability', 0.01)
        self.connection_probability = kwargs.get('connection_probability', 0.05)
        self.node_probability = kwargs.get('node_probability', 0.03)

        self.weight_randomization_factor = kwargs.get('weight_randomization_factor', 1.)
        self.weight_shift_factor = kwargs.get('weight_shift_factor', 1.)
        self.max_add_random_connection_tries = kwargs.get('max_add_random_connection_tries', 100)
        self.input_layer_nb = kwargs.get('input_layer_nb', 0)
        self.output_layer_nb = kwargs.get('output_layer_nb', 256)

        self.node_innovation_nb = 0
        self.minimal_genome = self.get_minimal_genome(self.input_size, self.output_size)
        self.nodes = {}


    def get_node(self, genome, connection):
        if not connection.innovation_nb in self.nodes:
            self.node_innovation_nb += 1

            nb_outputs = genome.get_nb_nodes_in_layer(connection.node_out.layer_nb)
            
            self.nodes[connection.innovation_nb] = NodeGene(self.node_innovation_nb, output=0,
               layer_nb=(connection.node_in.layer_nb + connection.node_out.layer_nb) / 2,
               index_in_layer=connection.node_in.index_in_layer * nb_outputs + connection.node_out.index_in_layer)

        return self.nodes[connection.innovation_nb].copy()
    

    def get_new_node(self, output, layer_nb, index_in_layer):
        self.node_innovation_nb += 1
        return NodeGene(self.node_innovation_nb, output=output, layer_nb=layer_nb, index_in_layer=index_in_layer)
    

    def get_minimal_genome(self, input_size, output_size):
        genome = Genome(self)

        inputs = [self.get_new_node(0, self.input_layer_nb, index_in_layer) for index_in_layer in range(input_size)]
        outputs = [self.get_new_node(0, self.output_layer_nb, index_in_layer) for index_in_layer in range(output_size)]
        
        for input_node in inputs:
            for output_node in outputs:
                connection = ConnectionGene(input_node, output_node, enabled=True,
                    weight_randomization_factor=self.weight_randomization_factor)
                
                input_node.connections.append(connection)
                genome.add_sorted_connection(connection)

        genome.nodes.extend(inputs)
        genome.nodes.extend(outputs)

        return genome

