import sys

def tsp_dp(n, cost):
    if n <= 1:
        return cost[0][0] if n == 1 else 0

    INF = sys.maxsize
    FULL = 1 << n
    fullMask = FULL - 1

    dp = [[INF] * n for _ in range(FULL)]
    dp[1][0] = 0

    for mask in range(1, FULL):
        for i in range(n):
            if not (mask & (1 << i)) or dp[mask][i] == INF:
                continue

            for j in range(n):
                if not (mask & (1 << j)):
                    nxt = mask | (1 << j)
                    new_cost = dp[mask][i] + cost[i][j]
                    if new_cost < dp[nxt][j]:
                        dp[nxt][j] = new_cost

    min_cycle_cost = INF
    for i in range(n):
        if dp[fullMask][i] != INF:
            min_cycle_cost = min(min_cycle_cost, dp[fullMask][i] + cost[i][0])

    return min_cycle_cost

def main():
    line = input().strip()
    while not line:  
        line = input().strip()
    n = int(line)

    cost = []
    for _ in range(n):
        cost.append([int(x) for x in input().split()])

    ans = tsp_dp(n, cost)
    print(ans)

if __name__ == "__main__":
    main()