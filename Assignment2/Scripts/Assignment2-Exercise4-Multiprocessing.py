#%%
import matplotlib.pyplot as plt
import multiprocessing 
from time import perf_counter
import numpy as np
from tqdm import tqdm

## Data as given the assignment 

def data1(n, sigma=10, rho=28, beta=8/3, dt=0.01, x=1, y=1, z=1):
    import numpy
    state = numpy.array([x, y, z], dtype=float)
    result = []
    for _ in range(n):
        x, y, z = state
        state += dt * numpy.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ])
        result.append(float(state[0] + 30))
    return result

## Algorithm which needs to be parallelized 

def alg2(data):
    if len(data) <= 1:
        return data
    else:
        split = len(data) // 2
        left = iter(alg2(data[:split])) # left data
        right = iter(alg2(data[split:])) # right data
        result = []
    # note: this takes the top items off the left and right piles
        left_top = next(left)
        right_top = next(right)
    # combining the left and right data
    while True:
        if left_top < right_top:
            result.append(left_top)
            try:
                left_top = next(left)
            except StopIteration:
            # nothing remains on the left; add the right + return
                return result + [right_top] + list(right)
        else:
            result.append(right_top)
            try:
                right_top = next(right)
            except StopIteration:
        # nothing remains on the right; add the left + return
                return result + [left_top] + list(left)


def parallel_alg2(data):
    if len(data) <= 1:
        return data
    else:
        split = len(data) // 2
        with multiprocessing.Pool() as p:   ##using multiprocessing to parallelize the two independent tasks
            [left, right] = p.map(
                alg2, [data[:split], data[split:]])
        left = iter(left)
        right = iter(right)
        # combining the left and right data
        result = []
        left_top = next(left)
        right_top = next(right)
    while True:
        if left_top < right_top:
            result.append(left_top)
            try:
                left_top = next(left)
            except StopIteration:
            # nothing remains on the left; add the right + return
                return result + [right_top] + list(right)
        else:
            result.append(right_top)
            try:
                right_top = next(right)
            except StopIteration:
        # nothing remains on the right; add the left + return
                return result + [left_top] + list(left)



if __name__ == '__main__':
    data_var = np.logspace(0, 23, base=2, dtype=int)
    duration = [] #time by the parallelized algo
    algo2d1time = [] #time by the unparallelized algo
    
    for n in tqdm(data_var):
        data_set = data1(n, sigma=10, rho=28, beta=8/3, dt=0.01, x=1, y=1, z=1)
        algo2d1_start = perf_counter()
        alg2(data_set)     ## time taken by the unparallelized algo
        algo2d1_stop= perf_counter()
        algo2d1time.append(algo2d1_stop - algo2d1_start)
        start_time = perf_counter()
        parallel_alg2(data_set)    #time taken by the parallelized algo
        stop_time = perf_counter()
        duration.append(stop_time - start_time)
        print(duration[-1],  algo2d1time[-1])
    plt.loglog(data_var, algo2d1time, label = "Unparallelized algo2")
    plt.loglog(data_var, duration, label = "Parallelized algo2")
    plt.title("Comparing two-process parallel implementation of algo2")
    plt.legend()
    plt.show()
# %%
