from neat.genome import Genome
import os.path
import time
from collections import deque


class GenomeSaving():
    def __init__(self, population, top=1, best_only=False, override=True, dir='', filenames=[]):
        self.top = top if top is not None else population
        self.override = override
        self.best_only = best_only
        self.best_fitness = 0
        self.filepaths = [os.path.join(dir, filename) for filename in filenames] if len(filenames) == top else \
        [os.path.join(dir, f'genome-{idx}.{Genome.FILE_EXT}' if self.override else f'genome-{idx}') for idx in range(top)]


    def __call__(self, dict_args):
        neat = dict_args['neat']
        generation = dict_args['generation']

        if self.best_only:
            best_genome = neat.genomes[0]

            if best_genome.fitness > self.best_fitness:
                self.best_fitness = best_genome.fitness
                best_genome.save(self.filepaths[0])
            
            return

        elif self.override:
            for idx in range(self.top):
                neat.genomes[idx].save(path=self.filepaths[idx])
            
            return

        for idx in range(self.top):
            neat.genomes[idx].save(path=self.filepaths[idx] + f'-gen-{generation}.{Genome.FILE_EXT}')


class FileLogger():
    def __init__(self, filepath, population, top=1):
        self.filepath = filepath
        self.top = top if top is not None else population
        self.data = deque()

    
    def __call__(self, dict_args):
        neat = dict_args['neat']
        generation = dict_args['generation']

        self.data.append(f'Gen {generation},{",".join([str(genome.fitness) for genome in neat.genomes])}\n')
        
        with open(self.filepath, 'w+') as file:
            file.writelines(self.data)



class TerminationCallback:
    def __init__(self):
        raise NotImplementedError()
    

    def __call__(self):
        raise NotImplementedError()


class GenerationTermination(TerminationCallback):
    def __init__(self, stop_at):
        self.stop_at = stop_at


    def __call__(self, dict_args):
        generation = dict_args['generation']
        return generation >= self.stop_at


class FitnessTermination(TerminationCallback):
    def __init__(self, termination_fitness, top=1):
        self.termination_fitness = termination_fitness
        self.top = top
    

    def __call__(self, dict_args):
        neat = dict_args['neat']

        for idx in range(self.top):
            if neat.genomes[idx].fitness < self.termination_fitness:
                return False
    
        return True
    

class TimeTermination(TerminationCallback):
    def __init__(self, hours, minutes=0, seconds=0):
        self.run_time = hours * 3600 + minutes * 60 + seconds

        self.start_time = time.perf_counter()


    def __call__(self, dict_args):
        return (time.perf_counter() - self.start_time) >= self.run_time

