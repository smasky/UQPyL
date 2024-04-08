import numpy as np

class Binary_GA():
    def __init__(self, evaluate, n_features, population_size=50, n_generations=100, crossover_rate=0.7, mutation_rate=0.01):
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_features = n_features
        self.evaluate = evaluate
    def initialize_population(self):
        return np.random.randint(2, size=(self.population_size, self.n_features))
    
    def select(self, fitnesses):
        total_fitness = sum(fitnesses)
        selection_probs = [f/total_fitness for f in fitnesses]
        return np.random.choice(range(self.population_size), size=self.population_size, replace=True, p=selection_probs)

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.n_features)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1, parent2

    def mutate(self, individual):
        for i in range(self.n_features):
            if np.random.rand() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        return individual
    
    def run(self):
        population = self.initialize_population()
        for generation in range(self.n_generations):
            fitnesses = [self.evaluate(individual) for individual in population]
            selected_indices = self.select(fitnesses)
            selected_population = population[selected_indices]
            offspring_population = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = selected_population[i], selected_population[i+1]
                child1, child2 = self.crossover(parent1, parent2)
                offspring_population.append(self.mutate(child1))
                offspring_population.append(self.mutate(child2))
            population = np.array(offspring_population)
        best_individual_index = np.argmin(fitnesses)
        return population[best_individual_index], min(fitnesses)