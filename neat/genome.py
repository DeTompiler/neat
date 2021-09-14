from collections import OrderedDict
import numpy as np
import pickle
import random
from neat.connection_gene import ConnectionGene


class Genome:
    ''' Genome class, contains the nodes and connection of the neural network
    '''
    
    FILE_EXT = 'gnc' # gnc - genome network configuration


    def __init__(self, input_keys, output_keys, output_activation):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.nodes = OrderedDict()
        self.connections = {}
        self.fitness = 0.0
        self.output_activation = output_activation


    def copy(self, copy_connections=True):
        genome = Genome(self.input_keys.copy(), self.output_keys.copy(), self.output_activation)
        genome.fitness = self.fitness

        for node_key in self.input_keys:
            genome.nodes[node_key] = self.nodes[node_key].copy()
        for node_key in self.output_keys:
            genome.nodes[node_key] = self.nodes[node_key].copy()

        if copy_connections:
            for conn in self.connections.values():
                node_in = genome.nodes.get(conn.node_in.innovation_nb)
                node_out = genome.nodes.get(conn.node_out.innovation_nb)

                if node_in is None:
                    node_in = conn.node_in.copy()
                    genome.nodes[node_in.innovation_nb]
                if node_out is None:
                    node_out = conn.node_out.copy()
                    genome.nodes[node_out.innovation_nb]
                
                copied_connection = conn.copy(node_in, node_out)
                node_in.connections.append(copied_connection)
                genome.connections[copied_connection.innovation_nb] = copied_connection

        return genome


    def assign_inputs(self, inputs):
        assert len(inputs) == len(self.input_keys)

        for node_key, input_value in zip(self.input_keys, inputs):
            self.nodes[node_key].output = input_value


    def predict(self, inputs):
        self.assign_inputs(inputs)

        for node in self.nodes.values():
            node.apply_activation()
            
            for conn in node.connections:
                if conn.enabled:
                    conn.node_out.output += float(conn.weight * node.output)

        outputs = self.output_activation(np.array([self.nodes[node_key].output for node_key in self.output_keys])).numpy()
        self.reset_nodes()

        return outputs


    def sort_nodes(self):
        self.nodes = OrderedDict(sorted(self.nodes.items(), key=lambda tupl:tupl[1].layer_nb))


    def reset_nodes(self):
        for node in self.nodes.values():
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

        self.nodes[node.innovation_nb] = node


    def add_nodes(self, nodes, copy=False, check_existence=False):
        for node in nodes:
            self.add_node(node, copy, check_existence)
        

    def add_connection(self, conn, add_nodes=False, check_existence=False):
        if check_existence and conn.innovation_nb in self.connections:
            return

        if add_nodes:
            node_in = self.nodes.get(conn.node_in.innovation_nb)
            node_out = self.nodes.get(conn.node_out.innovation_nb)

            if node_in is None:
                node_in = conn.node_in.copy()
                self.nodes[node_in.innovation_nb] = node_in
            if node_out is None:
                node_out = conn.node_out.copy()
                self.nodes[node_out.innovation_nb] = node_out
            
            conn = conn.copy(node_in, node_out) # copied connection overrides conn argument
            node_in.connections.append(conn)

        self.connections[conn.innovation_nb] = conn


    def randomize_weights(self, neat):
        for conn in self.connections.values():
            conn.random_weight(neat.config.weight_rand_factor)


    def mutate(self, neat):
        if random.random() < neat.config.toggle_conn_prob:
            self.toggle_random_connection()
        if random.random() < neat.config.shift_weight_prob:
            self.shift_random_weight(neat)
        if random.random() < neat.config.random_weight_prob:
            self.randomize_random_weight(neat)
        if random.random() < neat.config.add_conn_prob:
            self.add_random_connection(neat)
        if random.random() < neat.config.add_node_prob:
            self.add_random_node(neat)


    def random_node(self):
        return list(self.nodes.values())[random.randint(0, len(self.nodes) - 1)]
    

    def random_connection(self):
        return list(self.connections.values())[random.randint(0, len(self.connections) - 1)]


    def toggle_random_connection(self):
        conn = self.random_connection()
        conn.enabled = not conn.enabled


    def randomize_random_weight(self, neat):
        self.random_connection().random_weight(neat.config.weight_rand_factor)
    

    def shift_random_weight(self, neat):
        self.random_connection().shift_weight(neat.config.weight_shift_factor)
    

    def add_random_connection(self, neat):
        for try_idx in range(neat.config.add_conn_tries):
            node_in = self.random_node()
            node_out = self.random_node()

            if node_in.layer_nb == node_out.layer_nb:
                continue

            if node_in.layer_nb > node_out.layer_nb:
                node_in, node_out = node_out, node_in

            if not node_in.has_connection_to(node_out):
                if node_in.layer_nb > node_out.layer_nb:
                    node_in, node_out = node_out, node_in
                
                conn = ConnectionGene(node_in, node_out, enabled=True,
                    weight_rand_factor=neat.config.weight_rand_factor)
                
                node_in.connections.append(conn)
                self.connections[conn.innovation_nb] = conn
                
                return
    

    def add_random_node(self, neat):
        conn = self.random_connection()

        # new node
        new_node = neat.get_node(conn)
        
        # new connections
        conn1 = ConnectionGene(conn.node_in, new_node, weight=1, enabled=True)
        conn2 = ConnectionGene(new_node, conn.node_out, conn.weight, enabled=conn.enabled)
        
        # remove old connections
        conn.node_in.remove_connection_to(conn.node_out)

        conn.node_in.connections.append(conn1)
        new_node.connections.append(conn2)
        self.connections[conn1.innovation_nb] = conn1
        self.connections[conn2.innovation_nb] = conn2
        self.nodes[new_node.innovation_nb] = new_node

        del self.connections[conn.innovation_nb]


    def distance(self, neat, genome):
        # required when treating disjoint and excess genes differently, most likely
        # unnecessary and time consuming (may be removed in a future update)
        g1, g2 = self, genome
        max_innovation_nb1 = max([conn.innovation_nb for conn in g1.connections.values()])
        max_innovation_nb2 = max([conn.innovation_nb for conn in g2.connections.values()])

        if max_innovation_nb1 < max_innovation_nb2:
            g1, g2 = g2, g1
            max_innovation_nb1, max_innovation_nb2 = max_innovation_nb2, max_innovation_nb1

        similar_genes = 0
        disjoint_genes = 0
        excess_genes = 0
        weight_diff = 0

        for conn_key in g1.connections:
            if conn_key in g2.connections: # similar gene
                weight_diff += abs(g1.connections[conn_key].weight - g2.connections[conn_key].weight)
                similar_genes += 1
            elif conn_key > max_innovation_nb2: # excess gene
                excess_genes += 1
            else: # disjoint gene
                disjoint_genes += 1
            
        for conn_key in g2.connections:
            if conn_key not in g1.connections:
                disjoint_genes += 1

        avg_weight_diff = (weight_diff / similar_genes) if similar_genes > 0 else 0
        genes_nb = max(len(g1.connections), len(g2.connections))
        genes_nb = 1 if genes_nb < 20 else genes_nb

        return (neat.config.excess_distance_coefficient * excess_genes / genes_nb) \
        + (neat.config.disjoint_distance_coefficient * disjoint_genes / genes_nb) \
        + neat.config.weights_distance_coefficient * avg_weight_diff


    def crossover(self, neat, genome):
        g1, g2 = (self, genome) if self.fitness > genome.fitness else (genome, self)

        offspring = neat.base_genome.copy(copy_connections=False)

        for conn_key in g1.connections:
            if conn_key in g2.connections: # similar gene
                conn = g1.connections[conn_key] if random.random() > 0.5 else g2.connections[conn_key]
                offspring.add_connection(conn, add_nodes=True, check_existence=True)

            else: # disjoint or excess gene (which one does not matter for this function)
                offspring.add_connection(g1.connections[conn_key], add_nodes=True, check_existence=True)

        return offspring


    def __str__(self): 
        return '\n'.join([str(node) for node in self.nodes.values()])
    

    def save(self, path=f'genome.{FILE_EXT}'):
        with open(path, 'wb') as file:
            pickle.dump(self, file)