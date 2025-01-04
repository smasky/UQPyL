import numpy as np

def tournamentSelection( pop, K, N, *fitnesses ):
    """
    Perform K-tournament selection based on multiple fitness criteria.

    Parameters:
    - K: The number of candidates to compete in each tournament.
    - N: The number of selections to make.
    - fitnesses: The fitness values of the candidates (can be more than 2).

    Returns:
    - indices of the selected N solutions.
    """
    
    fitnessList = []
    
    for fitness in fitnesses:
        if isinstance(fitness, np.ndarray):
            fitness_2d = fitness.reshape(-1, 1) if fitness.ndim == 1 else fitness
            fitnessList += [fitness_2d[:, i] for i in range(fitness_2d.shape[1])]
            
    # Combine the fitness values and sort candidates based on all fitnesses in reverse order
    lexsort_keys = tuple(fitness.ravel() for fitness in reversed(fitnessList))
    
    # Rank based on the combined fitness values
    rankIndex = np.lexsort(lexsort_keys).reshape(-1, 1)
    rank = np.argsort(rankIndex, axis=0).ravel()

    # Perform K-tournament selection
    tourSelection = np.random.randint(0, high=fitnessList[0].shape[0], size=(N, K))

    # Find the winners based on rank within each tournament
    winner_indices_in_tournament = np.argmin(rank[tourSelection], axis=1).ravel()
    winners_original_order = tourSelection[np.arange(N), winner_indices_in_tournament]

    return pop[winners_original_order.ravel()]