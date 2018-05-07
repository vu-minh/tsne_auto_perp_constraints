import numpy as np


def gen_log_space_float(limit, n):
    start = np.log(1.0)
    stop = np.log(limit)
    res = np.logspace(start, stop, n, base=np.e, endpoint=True)
    return res


def gen_log_space_int(limit, n):
    """Generate sequence of integer with log-scale spacing
        link: https://stackoverflow.com/questions/12418234/logarithmically-spaced-integers
        args:
            limit: maximum target value  (the sequence is in [0, limit])
            n: number of points in sequence (len(result) == n)
    """
    result = [1]
    if n > 1:  # just a check to avoid ZeroDivisionError
        ratio = (float(limit) / result[-1]) ** (1.0 / (n - len(result)))
    while len(result) < n:
        next_value = result[-1] * ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value by artificially incrementing previous value
            result.append(result[-1] + 1)
            # recalculate the ratio so that the remaining values will scale correctly
            ratio = (float(limit) / result[-1]) ** (1.0 / (n - len(result)))
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x) - 1, result)), dtype=np.uint64)


if __name__ == '__main__':
    res = gen_log_space_float(limit=100, n=50)
    print(res)