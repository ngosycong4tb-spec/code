import sys

def knapsack(W, val, wt, n, memo):
    if n == 0 or W == 0:
        return 0

    if memo[n][W] != -1:
        return memo[n][W]

    pick = 0
    if wt[n - 1] <= W:
        pick = val[n - 1] + knapsack(W - wt[n - 1], val, wt, n - 1, memo)

    notPick = knapsack(W, val, wt, n - 1, memo)

    memo[n][W] = max(pick, notPick)
    return memo[n][W]


def main():
    data = sys.stdin.read().split()
    idx = 0

    n = int(data[idx]); idx += 1
    W = int(data[idx]); idx += 1

    wt = [0] * n
    val = [0] * n

    for i in range(n):
        wt[i] = int(data[idx]); idx += 1
        val[i] = int(data[idx]); idx += 1

    memo = [[-1] * (W + 1) for _ in range(n + 1)]
    result = knapsack(W, val, wt, n, memo)

    print(result)


main()