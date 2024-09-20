import numpy as np

def tournamentSelection(K, N, *fitnesses):
    """
    Perform K-tournament selection based on multiple fitness criteria.

    Parameters:
    - K: The number of candidates to compete in each tournament.
    - N: The number of selections to make.
    - fitnesses: The fitness values of the candidates (can be more than 2).

    Returns:
    - indices of the selected N solutions.
    """
    # Ensure all fitness values are numpy arrays and reshape them
    fitness_arrays = [np.array(fitness).reshape(-1, 1) for fitness in fitnesses]

    # Combine the fitness values and sort candidates based on all fitnesses in reverse order
    lexsort_keys = tuple(fitness.ravel() for fitness in reversed(fitness_arrays))
    
    # Rank based on the combined fitness values
    rankIndex = np.lexsort(lexsort_keys).reshape(-1, 1)
    rank = np.argsort(rankIndex, axis=0).ravel()

    # Perform K-tournament selection
    tourSelection = np.random.randint(0, high=fitness_arrays[0].shape[0], size=(N, K))

    # Find the winners based on rank within each tournament
    winner_indices_in_tournament = np.argmin(rank[tourSelection], axis=1).ravel()
    winners_original_order = tourSelection[np.arange(N), winner_indices_in_tournament]

    return winners_original_order.ravel()