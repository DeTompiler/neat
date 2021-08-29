from collections import deque
import pickle
from neat.connection_gene import ConnectionGene
from neat.genome import Genome
from neat.node_gene import NodeGene
from neat.species import Species
from tensorflow.keras.activations import sigmoid, relu
import random
import numpy as np


class Neat:
    ''' Neat class
        - input_size - the input size for each genome
        - output_size - the output size for each genome
        - node_innovation_nb - last used innovation number for a new node
        - nodes - dictionary mapping connection_innovation_nb -> NodeGene
    '''


    def __init__(self, input_size, output_size, population, **kwargs):
        self.input_size = input_size
        self.output_size = output_size
        self.population = population

        self.c1 = kwargs.get('c1', 1.0)
        self.c2 = kwargs.get('c2', 1.0)
        self.c3 = kwargs.get('c3', 0.4)

        self.output_activation = kwargs.get('output_activation', sigmoid)
        self.hidden_activation = kwargs.get('hidden_activation', relu)

        self.genome_distance_threshold = kwargs.get('genome_distance_threshold', 3.0)
        self.kill_worst = kwargs.get('kill_worst', 0.2)

        self.toggle_probability = kwargs.get('toggle_probability', 0.01)
        self.shift_probability = kwargs.get('shift_probability', 0.01)
        self.random_probability = kwargs.get('random_probability', 0.01)
        self.connection_probability = kwargs.get('connection_probability', 0.05)
        self.node_probability = kwargs.get('node_probability', 0.03)

        self.weight_randomization_factor = kwargs.get('weight_randomization_factor', 1.)
        self.weight_shift_factor = kwargs.get('weight_shift_factor', 1.)
        self.similar_fitness_range = kwargs.get('similar_fitness_range', 0.04)
        self.max_add_random_connection_tries = kwargs.get('max_add_random_connection_tries', 100)
        self.input_layer_nb = kwargs.get('input_layer_nb', 0)
        self.output_layer_nb = kwargs.get('output_layer_nb', 256)

        self.node_innovation_nb = 0
        self.nodes = {}
        self.base_genome = self.create_base_genome(self.input_size, self.output_size)
        self.genomes = [self.base_genome.copy() for idx in range(population)]
        self.species = deque()


    def get_node(self, connection):
        if not connection.innovation_nb in self.nodes:
            self.node_innovation_nb += 1

            self.nodes[connection.innovation_nb] = NodeGene(self.node_innovation_nb, output=0.0, activation=self.hidden_activation,
               layer_nb=(connection.node_in.layer_nb + connection.node_out.layer_nb) / 2)

        return self.nodes[connection.innovation_nb].copy()
    

    def get_new_node(self, output, activation, layer_nb):
        self.node_innovation_nb += 1
        return NodeGene(self.node_innovation_nb, output=output, activation=activation, layer_nb=layer_nb)
    

    def mutate_all(self):
        for genome in self.genomes:
            genome.mutate()
    

    def generate_species(self):
        self.species = deque()

        for genome in self.genomes:
            found = False

            for species in self.species:
                if species.add_genome(genome, check_compatibility=True):
                    found = True
                    break
            
            if not found:
                self.species.append(Species(genome))
    

    def kill_worst_genomes(self, sort_species=False):
        remaining_species = deque()

        for species in self.species:
            if sort_species:
                species.sort()

            species.kill(self.kill_worst, kill_in_neat=True)

            if species.size() > 0:
                remaining_species.append(species)

        self.species = remaining_species
    

    def reproduce(self):
        for idx in range(len(self.genomes), self.population):
            species = self.random_species()
            genome = species.breed()

            self.genomes.append(genome)
            species.add_genome(genome, check_compatibility=False)


    def adjust_genomes_fitness(self):
        for species in self.species:
            species.adjust_fitness()
            species.compute_fitness()
    

    def evolve(self):
        self.generate_species()
        self.adjust_genomes_fitness()
        self.kill_worst_genomes(sort_species=True)
        self.reproduce()
        self.mutate_all()

        
    def sort_genomes(self):
        self.genomes = sorted(self.genomes, key=lambda genome: genome.fitness, reverse=True)
    

    def forward_all(self, inputs, genomes_alive, single_input=False):
        predictions = np.empty(shape=(self.population, self.output_size))

        if single_input:
            for idx, genome in enumerate(self.genomes):
                if genomes_alive[idx]:
                    predictions[idx] = genome.forward(inputs)
        
        else:
            for idx, genome in enumerate(self.genomes):
                if genomes_alive[idx]:
                    predictions[idx] = genome.forward(inputs[idx])
            
        return predictions
    

    def reset_all_nodes(self):
        for genome in self.genomes:
            genome.reset_nodes()
        

    def reset_all_fitness(self):
        for genome in self.genomes:
            genome.reset_fitness()


    def best_genomes(self, top=1, sort=True, top_one_as_genome=False):
        if sort:
            self.sort_genomes()
        
        if top_one_as_genome and top == 1:
            return self.genomes[0]
        
        return [self.genomes[idx] for idx in range(top)]


    def random_species(self, fitness_prob=True):
        if not fitness_prob:
            return random.choice(self.species)
        
        return random.choices(self.species, weights=[species.fitness for species in self.species])[0]


    def load_genome(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)


    def create_base_genome(self, input_size, output_size):
        genome = Genome(self)

        inputs = [self.get_new_node(0.0, None, self.input_layer_nb) for index_in_layer in range(input_size)]
        outputs = [self.get_new_node(0.0, None, self.output_layer_nb) for index_in_layer in range(output_size)]
        
        for input_node in inputs:
            for output_node in outputs:
                connection = ConnectionGene(input_node, output_node, enabled=True,
                    weight_randomization_factor=self.weight_randomization_factor)
                
                input_node.connections.append(connection)
                genome.add_connection(connection)

        genome.add_nodes(inputs)
        genome.add_nodes(outputs)

        return genome

