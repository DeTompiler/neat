from neat.genome import Genome
import os.path
import time


class GenomeSaving():
    def __init__(self, population, top=1, best_only=False, override=True, dir='', filenames=[]):
        self.top = top if top is not None else population
        self.override = override
        self.best_only = best_only
        self.best_fitness = 0
        self.filepaths = [os.path.join(dir, filename) for filename in filenames] if len(filenames) == top else \
        [os.path.join(dir, f'genome-{idx}.{Genome.FILE_EXT}' if self.override else f'genome-{idx}') for idx in range(top)]


    def __call__(self, **kwargs):
        neat = kwargs['neat']
        generation = kwargs['generation']

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
        self.data = []

    
    def __call__(self, **kwargs):
        neat = kwargs['neat']
        generation = kwargs['generation']

        self.data.append(f'Gen {generation},{",".join([str(neat.genomes[idx].fitness) for idx in range(self.top)])}\n')
        
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


    def __call__(self, **kwargs):
        return kwargs['generation'] >= self.stop_at


class FitnessTermination(TerminationCallback):
    def __init__(self, termination_fitness, top=1):
        self.termination_fitness = termination_fitness
        self.top = top
    

    def __call__(self, **kwargs):
        neat = kwargs['neat']

        for idx in range(self.top):
            if neat.genomes[idx].fitness < self.termination_fitness:
                return False
    
        return True
    

class TimeTermination(TerminationCallback):
    def __init__(self, hours, minutes=0, seconds=0):
        self.run_time = hours * 3600 + minutes * 60 + seconds

        self.start_time = time.perf_counter()


    def __call__(self, **kwargs):
        return (time.perf_counter() - self.start_time) >= self.run_time


class EnvStopper():
    def __init__(self, max_step, differentiate_genomes=False):
        self.max_step = max_step
        self.differentiate_genomes = differentiate_genomes

    
    def __call__(self, step):
        return step >= self.max_step