import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from collections import deque
from neat.callbacks import TerminationCallback
import pickle
from neat.connection_gene import ConnectionGene
from neat.genome import Genome
from neat.node_gene import NodeGene
from neat.species import Species
import random
import numpy as np
from neat.callbacks import TerminationCallback
from neat.config import Config
import math


class Neat:
    ''' Neat class
        - input_size - the input size for each genome
        - output_size - the output size for each genome
        - node_innovation_nb - last used innovation number for a new node
        - nodes - dictionary mapping connection_innovation_nb -> NodeGene
    '''


    def __init__(self, input_size, output_size, population, config=Config(), init_population=True):
        self.input_size = input_size
        self.output_size = output_size
        self.population = population

        self.config = config

        self.node_innovation_nb = 0
        self.nodes = {}
        self.base_genome = self.create_base_genome(self.input_size, self.output_size)
        
        self.genomes = []
        self.species = deque()

        self.generation = 1
        
        if init_population:
            self.genomes = [self.base_genome.copy() for idx in range(population)]
            self.generate_species()
            self.mutate_genomes()


    def get_node(self, connection):
        if not connection.innovation_nb in self.nodes:
            self.node_innovation_nb += 1

            self.nodes[connection.innovation_nb] = NodeGene(self.node_innovation_nb, output=0.0, activation=self.config.hidden_activation,
               layer_nb=(connection.node_in.layer_nb + connection.node_out.layer_nb) / 2)

        return self.nodes[connection.innovation_nb].copy()
    

    def get_new_node(self, output, activation, layer_nb):
        self.node_innovation_nb += 1
        return NodeGene(self.node_innovation_nb, output=output, activation=activation, layer_nb=layer_nb)
    

    def mutate_genomes(self):
        for genome in self.genomes:
            genome.mutate()
    

    def new_representatives(self):
        representatives = deque()

        for species in self.species:
            representative = species.representative
            smallest_distance = None
            most_compatible_genome = None

            for genome in self.genomes:
                curr_distance = genome.distance(representative)
                
                if smallest_distance is None or curr_distance < smallest_distance:
                    smallest_distance = curr_distance
                    most_compatible_genome = genome
            
            self.genomes.remove(most_compatible_genome)
            representatives.append(most_compatible_genome)
        
        return representatives
    

    def reset_species(self, representatives):
        for representative, species in zip(representatives, self.species):
            species.reset(representative)


    def generate_species(self):
        new_representatives = self.new_representatives()
        self.reset_species(new_representatives)

        for genome in self.genomes:
            found = False

            for species in self.species:
                if species.add_genome(genome, check_compatibility=True):
                    found = True
                    break
            
            if not found:
                self.species.append(Species(genome))
        
        self.genomes.extend(new_representatives)
    

    def eliminate_genomes(self, sort_species=False):
        remaining_species = deque()

        for species in self.species:
            species.update_stagnation(self.generation)

            if self.generation - species.last_improvement >= self.config.max_stagnation:
                continue

            elif sort_species:
                species.sort()

            species.kill(1 - self.config.survivors, kill_in_neat=True)

            remaining_species.append(species)

        self.species = remaining_species
    

    def reproduce(self):
        if len(self.species) == 0:
            if self.config.reset_on_extinction:
                self.genomes = [self.base_genome.copy() for idx in range(self.population)]
                self.generate_species()
            
            return
        
        species_spawn = self.compute_species_spawn()

        for species, spawn in zip(self.species, species_spawn):
            for idx in range(spawn):
                genome = species.breed()
            
                self.genomes.append(genome)
                species.add_genome(genome, check_compatibility=False)


    def evolve(self):
        self.eliminate_genomes(sort_species=True)
        self.reproduce()

        if len(self.species) > 0:
            self.mutate_genomes()
            self.generate_species()

            return False
        
        return True

        
    def sort_genomes(self):
        self.genomes = sorted(self.genomes, key=lambda genome: genome.fitness, reverse=True)
    

    def forward_all(self, genomes, inputs, genomes_alive):
        predictions = np.empty(shape=(self.population, self.output_size))

        for idx, genome in enumerate(genomes):
            if genomes_alive[idx]:
                predictions[idx] = genome.forward(inputs[idx])
            
        return predictions
    

    def reset_all_nodes(self, genomes):
        for genome in genomes:
            genome.reset_nodes()
        

    def reset_all_fitness(self, genomes):
        for genome in genomes:
            genome.reset_fitness()

    
    def add_fitness(self, genomes, scores):
        for genome, score in zip(genomes, scores):
            genome.fitness += score


    def cycle_env(self, env, genomes, verbose, visualize):
        states = env.reset()

        done = False
        genomes_alive = [True for genome in genomes]

        while not done:
            if verbose == 2:
                self.log(None, None, final_log=False)

            if visualize:
                env.render()

            next_states, scores, genomes_alive, done = env.step(self.forward_all(genomes, states, genomes_alive))

            self.reset_all_nodes(genomes)
            states = next_states
        
        self.add_fitness(genomes, scores)
    

    def set_genomes(self, genomes):
        self.genomes = genomes

        for genome in self.genomes:
            genome.neat = self
        
        self.generate_species()


    def log(self, generation, top_fitness, clear_line=True, final_log=False):
        clear_log = '\033[2K' if clear_line else ''
        log_end = '\n' if final_log else '\r'

        log = clear_log + f'generation = {generation}, top_fitness = {top_fitness}'

        print(log, end=log_end)
    

    def handle_callbacks(self, callbacks):
        termination_callbacks = []
        other_callbacks = []

        termination_callbacks = []
        other_callbacks = []

        for callback in callbacks:
            if isinstance(callback, TerminationCallback):
                termination_callbacks.append(callback)
            else:
                other_callbacks.append(callback)     

        return termination_callbacks, other_callbacks  


    def compute_species_spawn(self):
        self.species = sorted(self.species, key=lambda species: species.fitness / species.size(), reverse=True)
        species_fitness = [species.fitness for species in self.species]
        fitness_sum = sum(species_fitness)

        assert fitness_sum > 0

        nb_spawns = self.population - len(self.genomes)
        species_spawns = [0] * len(self.species)

        spawn_sum = 0

        for idx, fitness in enumerate(species_fitness):
                species_spawns[idx] = int(nb_spawns * fitness / fitness_sum)
                spawn_sum += species_spawns[idx]

        if spawn_sum < nb_spawns:
            for idx in range(nb_spawns - spawn_sum):
                species_spawns[idx] += 1

        return species_spawns


    def fit(self, env, callbacks=[], verbose=0, visualize=False):
        termination_callbacks, other_callbacks = self.handle_callbacks(callbacks)
        
        callback_args = {'neat':self, 'generation':self.generation}
        terminate = False

        while not terminate:
            callback_args['generation'] = self.generation

            self.cycle_env(env, self.genomes, verbose, visualize)
            self.sort_genomes()
               
            if verbose == 1:
                self.log(self.generation, self.genomes[0].fitness, final_log=False)

            for callback in other_callbacks:
                callback(callback_args)

            for t_callback in termination_callbacks:
                if t_callback(callback_args):
                    terminate = True
                    break
            
            if terminate:
                break
            
            terminate = self.evolve()
            self.reset_all_fitness(self.genomes)
            self.generation += 1
        
        if visualize:
            env.close()
        
        if verbose > 0:
            self.log(self.generation, self.genomes[0].fitness, final_log=True)


    def test(self, env, genomes, callbacks=[], verbose=0, visualize=False):
        termination_callbacks, other_callbacks = self.handle_callbacks(callbacks)
        
        generation = 1
        callback_args = {'neat':self, 'generation':generation}
        terminate = False

        while not terminate:
            callback_args['generation'] = generation

            self.cycle_env(env, genomes, verbose, visualize)
            # self.sort_genomes()
               
            if verbose == 1:
                self.log(generation, genomes[0].fitness, final_log=False)

            for callback in other_callbacks:
                callback(callback_args)

            for t_callback in termination_callbacks:
                if t_callback(callback_args):
                    terminate = True
            
            if terminate:
                break
            
            self.reset_all_fitness(self.genomes)
            generation += 1
        
        if visualize:
            env.close()
        
        if verbose > 0:
            self.log(generation, self.genomes[0].fitness, final_log=True)



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
        

    def save_genomes(self, path):
        with open(path, 'wb') as file:
            for genome in self.genomes:
                genome.neat = None
                pickle.dump(genome, file)
                genome.neat = self


    def load_genomes(self, path):
        with open(path, 'rb') as file:
            return deque(pickle.load(file))
                

    def create_base_genome(self, input_size, output_size):
        genome = Genome(self)

        inputs = [self.get_new_node(0.0, None, self.config.input_layer_nb) for index_in_layer in range(input_size)]
        outputs = [self.get_new_node(0.0, None, self.config.output_layer_nb) for index_in_layer in range(output_size)]
        
        for input_node in inputs:
            for output_node in outputs:
                connection = ConnectionGene(input_node, output_node, enabled=True,
                    weight_rand_factor=self.config.weight_rand_factor)
                
                input_node.connections.append(connection)
                genome.add_connection(connection)

        genome.add_nodes(inputs)
        genome.add_nodes(outputs)

        return genome

