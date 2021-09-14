import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from neat.connection_gene import ConnectionGene
from neat.genome import Genome
from neat.node_gene import NodeGene
from neat.species import Species
from neat.callbacks import TerminationCallback, EnvStopper
from neat.config import Config
from neat.utils import Math



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
        self.species = []

        self.generation = 1
        
        if init_population:
            self.genomes = [self.base_genome.copy() for idx in range(population)]
            self.mutate_genomes()
            self.generate_species()


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
            genome.mutate(self)
    

    def new_representatives(self):
        representatives = []

        for species in self.species:
            representative = species.representative
            smallest_distance = None
            most_compatible_genome = None

            for genome in self.genomes:
                curr_distance = genome.distance(self, representative)
                
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
                if species.add_genome(self, genome, check_compatibility=True):
                    found = True
                    break
            
            if not found:
                self.species.append(Species(genome))
        
        self.genomes.extend(new_representatives)
    

    def eliminate_genomes(self, sort_species=False):
        remaining_species = []

        for species in self.species:
            species.update_stagnation(self.generation)

            if self.generation - species.last_improvement >= self.config.max_stagnation:
                continue

            elif sort_species:
                species.sort()

            species.kill(self, 1 - self.config.survivors, kill_in_neat=True)

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
                genome = species.breed(self)
            
                self.genomes.append(genome)
                species.add_genome(self, genome, check_compatibility=False)


    def evolve(self):
        self.eliminate_genomes(sort_species=True)
        self.reproduce()

        if len(self.species) > 0:
            self.mutate_genomes()
            self.generate_species()

            return False
        
        return True

        
    def sort_genomes(self, genomes):
        return sorted(genomes, key=lambda genome: genome.fitness, reverse=True)
    

    def forward_all(self, genomes, inputs, global_inputs, forward_genomes=None):
        predictions = np.empty(shape=(len(genomes), self.output_size))

        for idx, genome in enumerate(genomes):
            if forward_genomes is not None and not forward_genomes[idx]:
                continue
                            
            predictions[idx] = genome.forward(inputs if global_inputs else inputs[idx])

        return predictions

    
    def set_genomes_fitness(self, genomes, fitnesses):
        for genome, fitness in zip(genomes, fitnesses):
            genome.fitness = fitness
    

    def sort_genome_nodes(self, genomes):
        for genome in genomes:
            genome.sort_nodes()


    def run_env(self, env, genomes, env_stopper, global_inputs, visualize):
        states = env.reset()

        step = 0
        done = False
        genomes_alive = None

        while not done:
            if env_stopper is not None and env_stopper(step):
                if env_stopper.differentiate_genomes:
                    if not env_stopper.use_scores:
                        scores = [env_stopper.alive_fitness if genome_alive else 0 for genome_alive in genomes_alive]
                    self.set_genomes_fitness(genomes, scores)
                return True

            if visualize:
                env.render()

            next_states, scores, genomes_alive, done = env.step(self.forward_all(genomes, states, global_inputs, genomes_alive))
            states = next_states
            step += 1

        self.set_genomes_fitness(genomes, scores)
    
        return False


    def run_env_threaded(self, env, genomes, nb_threads, env_stopper, global_inputs, visualize, env_args):
        Env = env.__class__
        threads = []

        genomes_per_thread = len(genomes) // nb_threads
        complementary_genomes = len(genomes) - nb_threads * genomes_per_thread

        cut_idx = genomes_per_thread + complementary_genomes

        executor = ThreadPoolExecutor()

        for idx in range(1, nb_threads):
            threaded_env = Env(genomes_per_thread, *env_args)

            # thread (called future)
            threads.append(executor.submit(self.run_env, threaded_env, genomes[cut_idx:cut_idx + genomes_per_thread],
            env_stopper, global_inputs, False))
            cut_idx += genomes_per_thread

        terminate = self.run_env(Env(genomes_per_thread + complementary_genomes, *env_args),
            genomes[:genomes_per_thread + complementary_genomes], env_stopper, global_inputs, visualize)

        executor.shutdown(wait=True)

        if terminate:
            return True
        
        for thread in threads:
            if thread.result():
                return True
        
        return False

    
    def run_data(self, genomes, inputs, outputs, loss_function, global_inputs):
        genomes_fitness = np.zeros(shape=(len(genomes)))

        if type(outputs) is not np.ndarray:
            outputs = np.array(outputs)

        for input_, output in zip(inputs, outputs):
            predictions = self.forward_all(genomes, input_, global_inputs)
            losses = loss_function(output, predictions)

            if type(losses) is not np.ndarray:
                losses = np.array(losses)

            genomes_fitness -= losses
        
        max_abs_loss = np.abs(np.min(genomes_fitness))
        genomes_fitness += max_abs_loss + 1

        self.set_genomes_fitness(genomes, genomes_fitness)


    def set_genomes(self, genomes):
        self.genomes = genomes        
        self.generate_species()


    def log(self, generation, top_fitness=None, step=None, nb_species=None, clear_line=True, final_log=False):
        clear_log = '\033[2K' if clear_line else ''
        log_end = '\n' if final_log else '\r'

        step_log = f', step = {step}' if step is not None else ''
        species_log = f', species = {nb_species}' if nb_species is not None else ''
        top_fitness_log = f', top_fitness = {top_fitness}' if top_fitness is not None else ''

        log = clear_log + f'generation = {generation}{step_log}{species_log}{top_fitness_log}'

        print(log, end=log_end)
    

    def handle_callbacks(self, callbacks):
        termination_callbacks = []
        other_callbacks = []
        env_stopper = None

        termination_callbacks = []
        other_callbacks = []

        for callback in callbacks:
            if isinstance(callback, TerminationCallback):
                termination_callbacks.append(callback)
            elif isinstance(callback, EnvStopper):
                env_stopper = callback
            else:
                other_callbacks.append(callback)     

        return termination_callbacks, other_callbacks, env_stopper


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


    def fit_env(self, env, callbacks=[], threads=1, global_inputs=False, verbose=0, visualize=False, env_args=()):
        termination_callbacks, other_callbacks, env_stopper = self.handle_callbacks(callbacks)
        
        terminate = False

        while not terminate:
            self.sort_genome_nodes(self.genomes)

            if threads > 1:
                terminate = self.run_env_threaded(env, self.genomes, threads, env_stopper, global_inputs, visualize, env_args)
            else:
                terminate = self.run_env(env, self.genomes, env_stopper, global_inputs, visualize)

            self.genomes = self.sort_genomes(self.genomes)
               
            if verbose == 1:
                self.log(self.generation, top_fitness=self.genomes[0].fitness, nb_species=len(self.species), final_log=False)

            for callback in other_callbacks:
                callback(neat=self, generation=self.generation)

            for t_callback in termination_callbacks:
                if t_callback(neat=self, generation=self.generation):
                    terminate = True
                    break
            
            if terminate:
                break
            
            terminate = self.evolve()
            self.generation += 1
        
        if visualize:
            env.close()
        
        if verbose > 0:
            self.log(self.generation, top_fitness=self.genomes[0].fitness, nb_species=len(self.species), final_log=True)


    def fit_data(self, inputs, outputs, callbacks=[], loss_function=Math.mse, global_inputs=True, verbose=0):
        termination_callbacks, other_callbacks, env_stopper = self.handle_callbacks(callbacks)
        
        terminate = False

        while not terminate:
            self.sort_genome_nodes(self.genomes)
            self.run_data(self.genomes, inputs, outputs, loss_function, global_inputs)
            self.genomes = self.sort_genomes(self.genomes)
               
            if verbose == 1:
                self.log(self.generation, top_fitness=self.genomes[0].fitness, nb_species=len(self.species), final_log=False)

            for callback in other_callbacks:
                callback(neat=self, generation=self.generation)

            for t_callback in termination_callbacks:
                if t_callback(neat=self, generation=self.generation):
                    terminate = True
                    break
            
            if terminate:
                break
            
            terminate = self.evolve()
            self.generation += 1
        
        if verbose > 0:
            self.log(self.generation, top_fitness=self.genomes[0].fitness, nb_species=len(self.species), final_log=True)


    def test_env(self, env, genomes, callbacks=[], global_inputs=False, verbose=0, visualize=False):
        termination_callbacks, other_callbacks, env_stopper = self.handle_callbacks(callbacks)

        generation = 1
        terminate = False

        while not terminate:
            self.sort_genome_nodes(genomes)
            terminate = self.run_env(env, genomes, env_stopper, global_inputs, visualize)
            genomes = self.sort_genomes(genomes)
               
            if verbose == 1:
                self.log(generation, top_fitness=genomes[0].fitness, final_log=False)

            for callback in other_callbacks:
                callback(neat=self, generation=generation)

            for t_callback in termination_callbacks:
                if t_callback(neat=self, generation=generation):
                    terminate = True
            
            if terminate:
                break
            
            generation += 1
        
        if visualize:
            env.close()
        
        if verbose > 0:
            self.log(generation, top_fitness=genomes[0].fitness, final_log=True)
  
  
    def test_data(self, inputs, outputs, genomes, callbacks=[], loss_function=Math.mse, global_inputs=True, verbose=0):
        termination_callbacks, other_callbacks, env_stopper = self.handle_callbacks(callbacks)

        generation = 1
        terminate = False

        while not terminate:
            self.sort_genome_nodes(genomes)
            self.run_data(genomes, inputs, outputs, loss_function, global_inputs)
            genomes = self.sort_genomes(genomes)
               
            if verbose == 1:
                self.log(generation, top_fitness=genomes[0].fitness, final_log=False)

            for callback in other_callbacks:
                callback(neat=self, generation=generation)

            for t_callback in termination_callbacks:
                if t_callback(neat=self, generation=generation):
                    terminate = True
            
            if terminate:
                break
            
            generation += 1
        
        if verbose > 0:
            self.log(generation, top_fitness=genomes[0].fitness, final_log=True)



    def best_genomes(self, top=1, sort=True, top_one_as_genome=False):
        if sort:
            self.genomes = self.sort_genomes(self.genomes)
        
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
            pickle.dump(self.genomes, file)


    def load_genomes(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)
                

    def create_base_genome(self, input_size, output_size):
        input_nodes = [self.get_new_node(0.0, None, self.config.input_layer_nb) for index_in_layer in range(input_size)]
        output_nodes = [self.get_new_node(0.0, None, self.config.output_layer_nb) for index_in_layer in range(output_size)]

        input_keys = [input_node.innovation_nb for input_node in input_nodes]
        output_keys = [output_node.innovation_nb for output_node in output_nodes]

        genome = Genome(input_keys, output_keys, self.config.output_activation)

        for input_node in input_nodes:
            for output_node in output_nodes:
                conn = ConnectionGene(input_node, output_node, enabled=True,
                    weight_rand_factor=self.config.weight_rand_factor)
                
                input_node.connections.append(conn)
                genome.connections[conn.innovation_nb] = conn

        genome.add_nodes(input_nodes)
        genome.add_nodes(output_nodes)

        return genome

