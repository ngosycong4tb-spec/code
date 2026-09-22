import sys
import time
import random
import numpy as np

def read_cities_from_input():
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    cities = []
    has_coord_section = any("NODE_COORD_SECTION" in line for line in lines)
    in_coord_section = not has_coord_section

    for line in lines:
        line = line.strip()
        if not line or line == "EOF":
            if in_coord_section and has_coord_section:
                break
            continue
        if "NODE_COORD_SECTION" in line:
            in_coord_section = True
            continue
        if not in_coord_section:
            continue

        parts = line.split()
        try:
            if len(parts) >= 3:
                cities.append((float(parts[1]), float(parts[2])))
            elif len(parts) == 2:
                cities.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue

    return cities

def build_distance_matrix(cities):
    coords = np.array(cities)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    mat = np.sqrt(np.sum(diff**2, axis=-1))
    return mat

def tour_length(individual, dist_matrix):
    return sum(dist_matrix[individual[i-1]][individual[i]] for i in range(len(individual)))

def fitness(individual, dist_matrix):
    length = tour_length(individual, dist_matrix)
    return 1.0 / length if length > 0 else 0.0

def two_opt(individual, dist_matrix):
    n = len(individual)
    best_tour = individual[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                n1, n2 = best_tour[i-1], best_tour[i]
                n3, n4 = best_tour[j], best_tour[(j+1)%n]
                
                if dist_matrix[n1][n3] + dist_matrix[n2][n4] < dist_matrix[n1][n2] + dist_matrix[n3][n4] - 1e-5:
                    best_tour[i:j+1] = best_tour[i:j+1][::-1]
                    improved = True
    return best_tour

def greedy_initialization(dist_matrix, start_node):
    n = len(dist_matrix)
    unvisited = np.ones(n, dtype=bool) 
    unvisited[start_node] = False
    
    tour = [start_node]
    curr = start_node
    
    for _ in range(n - 1):
        dists = dist_matrix[curr].copy()
        dists[~unvisited] = np.inf 
        next_node = np.argmin(dists)
        
        tour.append(next_node)
        unvisited[next_node] = False
        curr = next_node
        
    return tour

def tournament_selection(population, fitness_scores, k=3):
    best_idx = random.randrange(len(population))
    best_fit = fitness_scores[best_idx]
    for _ in range(k - 1):
        idx = random.randrange(len(population))
        if fitness_scores[idx] > best_fit:
            best_idx = idx
            best_fit = fitness_scores[idx]
    return best_idx

def order_crossover(parent1, parent2):
    n = len(parent1)
    start = random.randrange(n)
    end = random.randrange(n)
    if start > end: start, end = end, start

    def make_child(p_main, p_other):
        child = [None] * n
        child[start:end + 1] = p_main[start:end + 1]
        used = set(child[start:end + 1])
        fill_values = (g for g in p_other if g not in used)
        for i in range(n):
            if child[i] is None:
                child[i] = next(fill_values)
        return child
    return make_child(parent1, parent2), make_child(parent2, parent1)

def swap_mutation(individual):
    ind = individual[:]
    i, j = random.randrange(len(ind)), random.randrange(len(ind))
    ind[i], ind[j] = ind[j], ind[i]
    return ind

def genetic_algorithm_tsp(dist_matrix, pop_size=40, iterations=1000, time_limit=1.5):
    start_time = time.time()
    n_cities = dist_matrix.shape[0]
    
    population = []
    for _ in range(pop_size):
        if time.time() - start_time > time_limit - 0.3:
            break
            
        start_node = random.randrange(n_cities)
        tour = greedy_initialization(dist_matrix, start_node)
        
        if len(population) < 2:
            tour = two_opt(tour, dist_matrix)
            
        population.append(tour)

    pop_size = len(population)
    
    if pop_size == 0:
        emergency_tour = greedy_initialization(dist_matrix, 0)
        return emergency_tour, tour_length(emergency_tour, dist_matrix)

    fitness_scores = [fitness(ind, dist_matrix) for ind in population]
    pop_data = list(zip(fitness_scores, population))
    pop_data.sort(key=lambda x: x[0], reverse=True)
    fitness_scores = [x[0] for x in pop_data]
    population = [x[1] for x in pop_data]

    best_length = tour_length(population[0], dist_matrix)
    best_tour = population[0]

    for gen in range(iterations):
        if time.time() - start_time > time_limit:
            break

        elitism_count = min(2, pop_size)
        new_population = [ind[:] for ind in population[:elitism_count]]

        while len(new_population) < pop_size:
            p1 = population[tournament_selection(population, fitness_scores, 3)]
            p2 = population[tournament_selection(population, fitness_scores, 3)]

            c1, c2 = order_crossover(p1, p2)
            
            if random.random() < 0.3: c1 = swap_mutation(c1)
            if random.random() < 0.3: c2 = swap_mutation(c2)

            if random.random() < 0.05: c1 = two_opt(c1, dist_matrix)
            
            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)

        population = new_population
        fitness_scores = [fitness(ind, dist_matrix) for ind in population]
        pop_data = list(zip(fitness_scores, population))
        pop_data.sort(key=lambda x: x[0], reverse=True)
        fitness_scores = [x[0] for x in pop_data]
        population = [x[1] for x in pop_data]

        current_best = tour_length(population[0], dist_matrix)
        if current_best < best_length:
            best_length = current_best
            best_tour = population[0]

    if time.time() - start_time < time_limit:
        best_tour = two_opt(best_tour, dist_matrix)
        best_length = tour_length(best_tour, dist_matrix)

    return best_tour, best_length

if __name__ == "__main__":
    cities = read_cities_from_input()
    if len(cities) < 2:
        sys.exit()

    dist_matrix = build_distance_matrix(cities)

    best_tour, best_length = genetic_algorithm_tsp(
        dist_matrix, pop_size=40, iterations=1000, time_limit=1.5
    )
    
    print(round(best_length, 3))