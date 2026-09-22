import sys
import random

def read_input():
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            tokens = f.read().split()
        if not tokens:
            return 0, 0, [], []
        n, W = int(tokens[0]), int(tokens[1])
        weights = [int(tokens[2 + 2 * i]) for i in range(n)]
        values = [int(tokens[3 + 2 * i]) for i in range(n)]
        return n, W, weights, values

    line = input().strip()
    while not line:
        line = input().strip()
    n, W = map(int, line.split())

    weights = []
    values = []
    for _ in range(n):
        w, v = map(int, input().split())
        weights.append(w)
        values.append(v)

    return n, W, weights, values

def knapsack_ga(n, W, weights, values, pop_size=100, generations=300, mutation_rate=None):
    if n == 0 or W == 0:
        return 0

    ratio_order = sorted(
        range(n),
        key=lambda i: values[i] / weights[i] if weights[i] > 0 else float('inf'),
        reverse=True
    )
    reverse_ratio_order = list(reversed(ratio_order))

    def repair_and_fill(individual):
        tot_w = sum(weights[i] for i in range(n) if individual[i])
        if tot_w > W:
            for i in reverse_ratio_order:
                if individual[i]:
                    individual[i] = 0
                    tot_w -= weights[i]
                    if tot_w <= W:
                        break
        for i in ratio_order:
            if not individual[i] and tot_w + weights[i] <= W:
                individual[i] = 1
                tot_w += weights[i]
        return individual

    def calculate_fitness(individual):
        tot_v = sum(values[i] for i in range(n) if individual[i])
        tot_w = sum(weights[i] for i in range(n) if individual[i])
        return tot_v if tot_w <= W else 0

    if mutation_rate is None:
        mutation_rate = max(0.01, 1.0 / n)

    population = []
    greedy_ind = repair_and_fill([0] * n)
    population.append(greedy_ind)

    total_all_weights = sum(weights)
    p_pick = min(0.5, W / total_all_weights) if total_all_weights > 0 else 0.5

    while len(population) < pop_size:
        ind = [1 if random.random() < p_pick else 0 for _ in range(n)]
        ind = repair_and_fill(ind)
        population.append(ind)

    fitness_scores = [calculate_fitness(ind) for ind in population]
    best_idx = max(range(pop_size), key=lambda i: fitness_scores[i])
    best_val = fitness_scores[best_idx]
    best_ind = population[best_idx][:]

    for gen in range(generations):
        def tournament_select():
            candidates = random.sample(range(pop_size), 3)
            best_cand = max(candidates, key=lambda i: fitness_scores[i])
            return population[best_cand]

        sorted_indices = sorted(range(pop_size), key=lambda i: fitness_scores[i], reverse=True)
        new_population = [population[sorted_indices[0]][:], population[sorted_indices[1]][:]]

        while len(new_population) < pop_size:
            p1 = tournament_select()
            p2 = tournament_select()

            child1 = [p1[i] if random.random() < 0.5 else p2[i] for i in range(n)]
            child2 = [p2[i] if random.random() < 0.5 else p1[i] for i in range(n)]

            for child in (child1, child2):
                for i in range(n):
                    if random.random() < mutation_rate:
                        child[i] = 1 - child[i]
                child = repair_and_fill(child)
                new_population.append(child)
                if len(new_population) >= pop_size:
                    break

        population = new_population[:pop_size]
        fitness_scores = [calculate_fitness(ind) for ind in population]

        current_max_idx = max(range(pop_size), key=lambda i: fitness_scores[i])
        if fitness_scores[current_max_idx] > best_val:
            best_val = fitness_scores[current_max_idx]
            best_ind = population[current_max_idx][:]

    return best_val

def main():
    n, W, weights, values = read_input()
    if n == 0:
        return
    best_val = knapsack_ga(n, W, weights, values)
    print(best_val)

if __name__ == "__main__":
    main()
