import numpy as np
from tqdm.auto import tqdm
import copy
from utils.leadlag import leadlag
from esig import tosig

class Organism:
    def __init__(self, n_points, pip, n_pips):
        self.n_points = n_points
        self.pip = pip
        self.n_pips = n_pips
        
        # Initialise
        self.randomise()

    def __add__(self, other):
        """Breed."""

        derivatives = []
        for derivative1, derivative2 in zip(self.derivatives, other.derivatives):
            if np.random.random() < 0.5:
                derivatives.append(derivative1)
            else:
                derivatives.append(derivative2)

        prices = np.r_[0., np.cumsum(derivatives)]
        path = leadlag(prices)

        o = Organism(self.n_points, self.pip, self.n_pips)
        o.derivatives = derivatives
        o.set_path(path)

        return o

    def random_derivative(self):
        r = self.pip * np.random.randint(-self.n_pips, self.n_pips)

        return r

    def randomise(self):
        self.derivatives = np.array([self.random_derivative() for _ in range(self.n_points - 1)])
        prices = np.r_[0., self.derivatives.cumsum()]

        path = leadlag(prices)
        self.set_path(path)


    def mutate(self, prob=0.1):
        for i in range(len(self.derivatives)):
            if np.random.random() < prob:
                self.derivatives[i] = self.random_derivative()

        prices = np.r_[0., np.cumsum(self.derivatives)]
        path = leadlag(prices)
        self.set_path(path)

    def set_path(self, path):
        self.path = path

    def logsignature(self, order):
        return tosig.stream2logsig(self.path, order)

    def loss(self, sig, order):
        diff = np.abs((sig - self.logsignature(order)) / sig)
        diff /= 1 + np.arange(len(sig))
        return np.mean(diff)

class Population:
    def __init__(self, n_organisms, n_points, pip, n_pips):
        self.n_points = n_points
        self.pip = pip
        self.n_pips = n_pips
        self.n_organisms = n_organisms

        self.organisms = [Organism(n_points, pip, n_pips) for _ in range(n_organisms)]

    def fittest(self, sig, p, order):
        n = int(len(self.organisms) * p)
        return sorted(self.organisms, key=lambda o: o.loss(sig, order))[:n]
        
    def evolve(self, sig, p, order, mutation_prob=0.1):
        parents = self.fittest(sig, p, order)
        new_generation = copy.deepcopy(parents)

        while len(new_generation) != self.n_organisms:
            i = j = 0
            while i == j:
                i, j = np.random.choice(range(len(parents)), size=2)
                parent1, parent2 = parents[i], parents[j]

            child = parent1 + parent2
            child.mutate(prob=mutation_prob)
            
            new_generation.append(copy.deepcopy(child))

        self.organisms = new_generation

        # Return loss
        return new_generation[0].loss(sig, order)

def train(sig, order, n_iterations, n_organisms, n_points, pip, n_pips,
          top_p=0.1, mutation_prob=0.1):
    population = Population(n_organisms, n_points, pip, n_pips)
    pbar = tqdm(range(n_iterations))

    for _ in pbar:
        loss = population.evolve(sig, p=top_p, order=order, mutation_prob=mutation_prob)
        pbar.set_description(f"Loss: {loss}")
        pbar.refresh()

        if loss == 0.:
            break

    return population.fittest(sig, p=top_p, order=order)[0].path[::2, 1], loss

        
        