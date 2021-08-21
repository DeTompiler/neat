from collections import deque, OrderedDict
import numpy as np
from src.rl.neat.connection_gene import ConnectionGene
from src.rl.neat.node_gene import NodeGene


class Genome:
    ''' Genome class, contains the nodes and connection of the neural network
    '''

    def __init__(self, neat):
        self.neat = neat
        self.layers = OrderedDict()
        self.nodes = deque()
        self.connections = deque()


    def copy(self):
        genome = Genome(self.neat)
        
        for layer_nb, layer in self.layers.items():
            copied_layer = deque()

            for node in layer:
                copied_node = node.copy()
                genome.nodes.append(copied_node)
                copied_layer.append(copied_node)

            genome.layers[layer_nb] = copied_layer

        for connection in self.connections:
            copied_connection = connection.copy()

            genome.connections.append(copied_connection)
            copied_connection.node_in.connections.append(copied_connection)
        
        return genome


    def reset(self):
        for node in self.nodes:
            node.output = 0

    

    def add_node(self, node):
        self.nodes.append(node)

        if not node.layer_nb in self.layers:
            self.layers[node.layer_nb] = deque([node])
        else:
            self.layers[node.layer_nb].append(node)
    

    def add_nodes(self, nodes):
        for node in nodes:
            self.add_node(node)
        

    def add_connection(self, connection):
        for idx in range(len(self.connections)):
            if self.connections[idx].innovation_nb < connection.innovation_nb:
                self.connections.insert(idx+1, connection)
                return
        
        self.connections.append(connection)


    def randomize_weights(self):
        for connection in self.connections:
            connection.random_weight(self.neat.weight_randomization_factor)


    def mutate(self):
        probabilities = np.random.rand(5) # 5 - number of possible mutations

        if probabilities[0] < self.neat.toggle_probability:
            self.toggle_random_connection()
        if probabilities[1] < self.neat.shift_probability:
            self.shift_random_weight()
        if probabilities[2] < self.neat.random_probability:
            self.randomize_random_weight()
        if probabilities[3] < self.neat.connection_probability:
            self.add_random_connection()
        if probabilities[4] < self.neat.node_probability:
            self.add_random_node()


    def random_node(self):
        return self.nodes[np.random.randint(0, len(self.nodes))]
    

    def random_connection(self):
        return self.connections[np.random.randint(0, len(self.connections))]


    def toggle_random_connection(self):
        connection = self.random_connection()
        connection.enabled = not connection.enabled


    def randomize_random_weight(self):
        self.random_connection().random_weight(self.neat.weight_randomization_factor)
    

    def shift_random_weight(self):
        self.random_connection().shift_weight(self.neat.weight_shift_factor)
    

    def add_random_connection(self):
        for try_idx in range(self.neat.max_add_random_connection_tries):
            node_in = self.random_node()
            node_out = self.random_node()

            if (not node_in.has_connection_to(node_out)) and node_in.layer_nb != node_out.layer_nb:
                if node_in.layer_nb > node_out.layer_nb:
                    node_in, node_out = node_out, node_in
                
                self.connections.append(ConnectionGene(node_in, node_out, enabled=True,
                    weight_randomization_factor=self.neat.weight_randomization_factor))
                
                return
    

    def add_random_node(self):
        connection = self.random_connection()

        new_node = self.neat.get_node(self, connection)

        self.connections.append(ConnectionGene(connection.node_in, new_node, weight=1, enabled=True))
        self.connections.append(ConnectionGene(new_node, connection.node_out, connection.weight, enabled=connection.enabled))
        
        self.nodes.append(new_node)
        self.connections.remove(connection)


    def distance(self, genome):
        g1, g2 = self, genome if self.connections[-1].innovation_nb > genome.connections[-1].innovation_nb else genome, self

        idx1 = 0
        idx2 = 0
        len1 = len(g1.connections)
        len2 = len(g2.connections)

        similar_genes = 0
        disjoint_genes = 0
        weight_diff = 0

        while(idx1 < len1 and idx2 < len2):
            con1 = g1.connections[idx1]
            con2 = g2.connections[idx2]

            if con1.innovation_nb == con2.innovation_nb:
                similar_genes += 1
                weight_diff += abs(con1.weight - con2.weight)
                idx1 += 1
                idx2 += 1

            elif con1.innovation_nb > con2.innovation_nb:
                disjoint_genes += 1
                idx2 += 1
            else:
                disjoint_genes += 1
                idx1 += 1    

        avg_weight_diff = weight_diff / similar_genes
        excess_genes = len(g1.connections) - idx1

        genes_nb = max(len(g1.connections), len(g2.connections))
        genes_nb = 1 if genes_nb < 20 else genes_nb

        return (self.neat.c1 * excess_genes / genes_nb) + (self.neat.c2 * disjoint_genes / genes_nb) + self.neat.c3 * avg_weight_diff


    def crossover(self, gemone):
        pass