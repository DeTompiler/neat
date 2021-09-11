import math
import random


class Species:
    def __init__(self, representative):
        self.representative = representative
        self.genomes = [representative]
        self.fitness = None
        self.max_fitness = None
        self.last_improvement = None

    
    def size(self):
        return len(self.genomes)


    def compatible(self, neat, genome):
        return self.representative.distance(neat, genome) <= neat.config.genome_distance_threshold
    

    def add_genome(self, neat, genome, check_compatibility=True):
        if genome is self.representative:
            return True

        elif check_compatibility and not self.compatible(neat, genome):
            return False
        
        self.genomes.append(genome)
        return True
    

    def sort(self):
        self.genomes = sorted(self.genomes, key=lambda genome: genome.fitness, reverse=True)


    def kill(self, neat, precentage, kill_in_neat=False):
        size = math.ceil(len(self.genomes) * precentage)
        
        if len(self.genomes) - size < neat.config.min_species_population:
            size = len(self.genomes) - neat.config.min_species_population

        if kill_in_neat:
            for idx in range(size):
                genome = self.genomes.pop()
                neat.genomes.remove(genome)
            
            return
        
        for idx in range(size):
            genome = self.genomes.pop()
        

    def random_genome(self, fitness_prob=True):
        if not fitness_prob:
            return random.choice(self.genomes)

        return random.choices(self.genomes, weights=[genome.fitness for genome in self.genomes])[0]
    

    def breed(self, neat):
        return self.random_genome(fitness_prob=True).crossover(neat, self.random_genome(fitness_prob=True))

    
    def adjust_fitness(self):
        species_size = self.size()

        for genome in self.genomes:
            genome.fitness /= species_size


    def compute_fitness(self):
        self.fitness = 0.0

        for genome in self.genomes:
            self.fitness += genome.fitness
    

    def update_stagnation(self, generation):
        self.adjust_fitness()
        self.compute_fitness()

        if self.max_fitness is None or self.fitness > self.max_fitness:
            self.max_fitness = self.fitness
            self.last_improvement = generation
        

    def reset(self, representative=None):
        self.representative = self.random_genome(fitness_prob=False) if representative is None else representative
        self.genomes = [self.representative]
        self.fitness = None