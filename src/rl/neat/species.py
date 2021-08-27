from collections import deque
import math


class Species:
    def __init__(self, representative):
        self.representative = representative
        self.genomes = deque([representative])

    
    def size(self):
        return len(self.genomes)


    def compatible(self, genome):
        return self.representative.distance(genome) <= self.representative.neat.genome_distance_threshold
    

    def add_genome(self, genome, check_compatibility=True):
        if check_compatibility and not self.compatible(genome):
            return False
        
        self.genomes.append(genome)
        return True
    

    def sort(self):
        self.genomes = sorted(self.genomes, key=lambda genome: genome.fitness, reverse=True)


    def kill(self, precentage):
        size = math.ceil(len(self.genomes) * precentage)

        for idx in range(size):
            self.genomes.pop()