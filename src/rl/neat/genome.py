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
        self.fitness = 0.0


    def copy(self):
        genome = Genome(self.neat)
        genome.fitness = self.fitness

        input_layer = self.layers[self.neat.input_layer_nb]
        output_layer = self.layers[self.neat.output_layer_nb]

        for node in input_layer:
            genome.add_node(node.copy())
        for node in output_layer:
            genome.add_node(node.copy())

        for connection in self.connections:
            node_in = genome.get_node(connection.node_in.innovation_nb)
            node_out = genome.get_node(connection.node_out.innovation_nb)

            if node_in is None:
                node_in = connection.node_in.copy()
                genome.add_node(node_in)
            if node_out is None:
                node_out = connection.node_out.copy()
                genome.add_node(node_out)
            
            copied_connection = connection.copy(node_in, node_out)
            node_in.connections.append(copied_connection)
            genome.add_connection(copied_connection)

        return genome


    def assign_inputs(self, inputs):
        input_layer = self.layers[self.neat.input_layer_nb]
        assert len(inputs) == len(input_layer)

        for node, input in zip(input_layer, inputs):
            node.output = input


    def layer_to_np_array(self, layer_nb):
        layer = self.layers[layer_nb]     
        return np.array([node.output for node in layer]) 
    

    def forward(self, inputs):
        self.assign_inputs(inputs)

        for layer in self.layers.values():
            for node in layer:
                node.apply_activation()

                for connection in node.connections:
                    if connection.enabled:
                        connection.node_out.output += connection.weight * node.output

        output_layer = self.layers[self.neat.output_layer_nb]
        if len(output_layer) == 1:
            return self.neat.output_activation(output_layer[0].output).numpy()
        
        return self.neat.output_activation(self.layer_to_np_array(self.neat.output_layer_nb)).numpy()


    def reset(self):
        self.fitness = 0.0

        for node in self.nodes:
            node.output = 0.0


    def get_node(self, innovation_nb):
        for node in self.nodes:
            if node.innovation_nb == innovation_nb:
                return node
        
        return None

    
    def node_exists(self, innovation_nb):
        return self.get_node(innovation_nb) is not None


    def connection_exists(self, innovation_nb):
        for connection in self.connections:
            if connection.innovation_nb == innovation_nb:
                return True
        
        return False


    def add_node(self, node, copy=False, check_existence=False):
        if check_existence and self.node_exists(node.innovation_nb):
            return

        if copy:
            node = node.copy()

        self.nodes.append(node)

        if not node.layer_nb in self.layers:
            layer_nbs = list(self.layers.keys())
            self.layers[node.layer_nb] = deque([node])

            for layer_nb in layer_nbs:
                if layer_nb > node.layer_nb:
                    self.layers.move_to_end(layer_nb)

        else:
            self.layers[node.layer_nb].append(node)
    

    def add_nodes(self, nodes, copy=False, check_existence=False):
        for node in nodes:
            self.add_node(node, copy, check_existence)
        

    def add_connection(self, connection, add_nodes=False, copy=False, check_existence=False):
        if check_existence and self.connection_exists(connection.innovation_nb):
            return

        if copy:
            connection = connection.copy()
        
        elif add_nodes:
            node_in = self.get_node(connection.node_in.innovation_nb)
            node_out = self.get_node(connection.node_out.innovation_nb)

            if node_in is None:
                node_in = connection.node_in.copy()
                self.add_node(node_in)
            if node_out is None:
                node_out = connection.node_out.copy()
                self.add_node(node_out)
            
            connection = connection.copy(node_in, node_out)
            node_in.connections.append(connection)

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

            if node_in.layer_nb == node_out.layer_nb:
                continue

            if node_in.layer_nb > node_out.layer_nb:
                node_in, node_out = node_out, node_in

            if not node_in.has_connection_to(node_out):
                if node_in.layer_nb > node_out.layer_nb:
                    node_in, node_out = node_out, node_in
                
                connection = ConnectionGene(node_in, node_out, enabled=True,
                    weight_randomization_factor=self.neat.weight_randomization_factor)
                
                node_in.connections.append(connection)
                self.add_connection(connection)
                
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


    def crossover(self, genome):
        g1, g2 = self, genome if self.fitness >= genome.fitness else genome, self
        similar_fitness = self.similar_fitness(genome)

        idx1, idx2 = 0, 0
        len1, len2 = len(g1.connections), len(g2.connections)

        offspring = self.neat.base_genome.copy()

        while idx1 < len1 and idx2 < len2:
            con1 = g1.connections[idx1]
            con2 = g2.connections[idx2]

            if con1.innovation_nb == con2.innovation_nb and not offspring.connection_exists(con1.innovation_nb):
                con = con1 if np.random.rand() > 0.5 else con2

                offspring.add_connection(con, add_nodes=True, copy=False, check_existence=True)

                idx1 += 1
                idx2 += 1

            elif con1.innovation_nb > con2.innovation_nb:
                # disjoint of g2
                if similar_fitness:
                    offspring.add_connection(con2, add_nodes=True, copy=False, check_existence=True)
                
                idx2 += 1
            else:
                # disjoint of g1
                offspring.add_connection(con1, add_nodes=True, copy=False, check_existence=True)
                idx1 += 1

        
        for idx in range(idx1, len1):
            con = g1.connections[idx]
            offspring.add_connection(con, add_nodes=True, copy=False, check_existence=True)
        
        if similar_fitness:
            for idx in range(idx2, len2):
                con = g2.connections[idx]
                offspring.add_connection(con, add_nodes=True, copy=False, check_existence=True)


        return offspring


    def similar_fitness(self, genome):
        if self.fitness == genome.fitness:
            return True
        
        # g1 has better fitness than g2
        g1, g2 = self, genome if self.fitness > genome.fitness else genome, self

        return g2.fitness > (g1.fitness - g1.fitness * self.neat.similar_fitness_range)


    def __str__(self):
        string = ''

        for layer_nb in self.layers:
            layer = self.layers[layer_nb]
            string += f'layer {layer_nb}:\n'

            for node in layer:
                string += f'\t{node}\n'
    
        return string