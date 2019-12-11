import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()


class Ant:
    
    
    def __init__(self, alpha, beta):
        
        # Should these be fixed or callable like in AntSystem?
        # Should ants modify their movement policy during a run?
        self.alpha = alpha
        self.beta = beta
        
        
    def walk(self, affinities, pheromones, allowed_locations, first_location, close_loop=True, greedy=False):
        
        remaining_choices = allowed_locations.copy()

        self.path = [first_location]
        this_location = first_location
        remaining_choices = remaining_choices[remaining_choices!=this_location]

        while len(remaining_choices) > 0:

            relevant_affinities = affinities[this_location, remaining_choices]
            relevant_pheromones = pheromones[this_location, remaining_choices]

            if np.all(relevant_affinities==0) or np.all(relevant_pheromones==0):
                # We reach this branch if the ant got cornered before the full tour was completed.
                self.path = []
                return self.path
            
            next_location_index = self.choose(relevant_affinities, relevant_pheromones, greedy=greedy)
            next_location = remaining_choices[next_location_index]
            
            self.path.append(next_location)
            
            this_location = next_location
            remaining_choices = remaining_choices[remaining_choices!=this_location]
            
        if close_loop:
            self.path.append(first_location)
            
        return self.path
                
    
    def choose(self, affinities, pheromones, greedy=False):
        if greedy:
            return pheromones.argmax()
        unnormalized_transition_probabilities = np.power(pheromones, self.alpha) \
                                                * np.power(affinities, self.beta)
        transition_probabilities = unnormalized_transition_probabilities \
                                   / unnormalized_transition_probabilities.sum()
        return rng.choice(len(affinities), p=transition_probabilities)
    
    
class AntSystem:
    
    
    def __init__(self, distances, init_pheromones=None, alpha=1, beta=2, rho=0.2):
        
        self.N = distances.shape[0]
        
        # This class allows for callable functions for alpha/beta/rho
        # in case you want to e.g. sample from a distribution.
        # The function should take no arguments.
        self.alpha = alpha if callable(alpha) else (lambda: alpha)
        self.beta = beta if callable(beta) else (lambda: beta)
        self.rho = rho if callable(rho) else (lambda: rho)
        
        self.distances = distances
        self.affinities = np.divide(1, distances, where=distances!=0, out=np.zeros_like(distances))
        
        # The "pheromones" which guide the ants. The diagonal should be all zeros.
        if init_pheromones is None:
            self.pheromones = np.ones_like(self.affinities) / self.rho()
            self.pheromones[np.where(self.affinities==0)] = 0
        else:
            self.pheromones = init_pheromones
        
        self.best_path, self.best_path_length = self.get_greedy_path()
        
        
    def solve(self, allowed_locations=None, start_location=None, close_loop=True):
        ant = Ant(self.alpha(), self.beta())
        if allowed_locations is None:
            allowed_locations = np.arange(self.N)
        if start_location is None:
            start_location = rng.choice(allowed_locations)
        return ant.walk(self.affinities, self.pheromones, 
                        allowed_locations, start_location, close_loop=close_loop)
    
    
    def solven(self, n_ants, allowed_locations=None, start_location=None, close_loop=True):
        return [self.solve(allowed_locations=allowed_locations, 
                           start_location=start_location, 
                           close_loop=close_loop)
                for _ in range(n_ants)]
        
        
    def update(self, paths, symmetric=True):
        """
        Performs one update of the pheromone matrix.
        """
            
        pheromone_update_matrix = np.zeros_like(self.pheromones)
        
        for path in paths:
            
            if len(path) > 0:
                this_pheromone_update_matrix, path_length = self.make_update_matrix(path,
                                                                                    symmetric=symmetric,
                                                                                    return_length=True)
                pheromone_update_matrix += this_pheromone_update_matrix
                
                if self.best_path_length > path_length:
                    self.best_path_length = path_length
                    self.best_path = path
                
        self.pheromones *= 1 - self.rho()
        self.pheromones += pheromone_update_matrix
        
        
    def plot(self, data):
        
        fig, ax = plt.subplots(figsize=(10,10))
        normalized_transition_matrix = self.pheromones / self.pheromones.sum(axis=1, keepdims=True)
        for p1,p2 in zip(*np.where(normalized_transition_matrix > 1/self.N)):
            alpha = np.power(normalized_transition_matrix[p1,p2], 1/np.power(self.N, 0.2))
            lw = 4 * normalized_transition_matrix[p1,p2]
            ax.plot(data[[p1,p2],0], data[[p1,p2],1], color='k', 
                    alpha=alpha, lw=lw, zorder=1)
        ax.scatter(*(data.T), color='r', marker='*', s=50, zorder=2)
        return fig, ax

            
    def make_update_matrix(self, path, symmetric=True, return_length=False):
        length = self.get_path_length(path)
        update_matrix = np.zeros_like(self.pheromones)
        for p1,p2 in zip(path[:-1], path[1:]):
            update_matrix[p1,p2] += 1
        if symmetric:
            update_matrix += update_matrix.T
        if return_length:
            return update_matrix / length, length
        return update_matrix / length
        
        
    def get_path_length(self, path):
        if len(path) == 0:
            return np.inf  # to catch when path is result of aborted run
        cumulative_length = 0
        for p1, p2 in zip(path[:-1], path[1:]):
            cumulative_length += self.distances[p1, p2]
        return cumulative_length
    
    
    def get_greedy_path(self, start_location=None, close_loop=True):
        ant = Ant(self.alpha(), self.beta())
        if start_location is None:
            start_location = rng.choice(self.N)
        ant.walk(self.affinities, self.pheromones, 
                 np.arange(self.N, dtype=int), start_location, 
                 close_loop=close_loop, greedy=True)
        return ant.path, self.get_path_length(ant.path)
    

class MaxMinAntSystem(AntSystem):
    
    # based on https://doi.org/10.1016/S0167-739X(00)00043-1
    
    def __init__(self, distances, alpha=1, beta=2, rho=0.5):
        super().__init__(distances, init_pheromones=None, alpha=alpha, beta=alpha, rho=rho)
        _, max_pheromone = self.pheromone_bounds()
        self.pheromones = np.ones_like(distances) * max_pheromone
        self.pheromones[np.where(self.affinities==0)] = 0
        
        
    def update(self, paths, symmetric=True):
        
        path_lengths = np.array([self.get_path_length(path) for path in paths])
        shortest_path_index = path_lengths.argmin()  # winner-take-all
        shortest_path_length = path_lengths[shortest_path_index]
        pheromone_update_matrix = self.make_update_matrix(paths[shortest_path_index],
                                                                       symmetric=symmetric)
        if self.best_path_length > shortest_path_length:
            self.best_path_length = shortest_path_length
            self.best_path = paths[shortest_path_index]

        self.pheromones *= 1 - self.rho()
        self.pheromones += pheromone_update_matrix
        self.pheromones = np.clip(self.pheromones, *self.pheromone_bounds())
        self.pheromones[np.where(self.affinities==0)] = 0
        
        
    def make_update_matrix(self, path, symmetric=True, return_length=False):
        length = self.get_path_length(path)
        update_matrix = np.zeros_like(self.pheromones)
        for p1,p2 in zip(path[:-1], path[1:]):
            update_matrix[p1,p2] += 1
        if symmetric:
            update_matrix += update_matrix.T
        if return_length:
            return update_matrix, length
        return update_matrix
    
    
    def pheromone_bounds(self):
        max_pheromone = 1 / (self.rho() * self.best_path_length)
        N = self.affinities.shape[0]
        nth_root_one_twentieth = np.power(0.05, 1/N)
        min_pheromone = max_pheromone * (1 - nth_root_one_twentieth) / ((N / 2 - 1) * nth_root_one_twentieth)
        return min_pheromone, max_pheromone
