import numpy as np
import h5py
from tqdm import tqdm
import multiprocessing as mp

grid = [
    [0, 4, 8, 12],
    [1, 5, 9, 13],
    [2, 6, 10, 14],
    [3, 7, 11, 15]
]

grid_flat = list(range(0, 16))
grid_points = []
for x in range(0, 4):
    for y in range(0, 4):
        grid_points.append((x, y))
nodes = 16
half_nodes = 8

initial_partition = ((0, 0))

def partition_point_to_digit(points):
    return [
        grid[point[0]][point[1]] for point in points
    ]
def partition_digit_inversion(digits):
    return np.setdiff1d([i for i in range(half_nodes)], digits)

def grow_partitions(partition):
    new_partitions = []
    allowed_nodes = grid_points
    forbidden_nodes = partition
    allowed_nodes = [
        point for point in allowed_nodes 
        if not point in forbidden_nodes]
    for node in allowed_nodes:
        new_partition = partition.copy()
        new_partition.append(node)
        new_partitions.append(new_partition)
    return new_partitions


if __name__ == "__main__":
    pool = mp.Pool(processes=8)
    partition_length = 1
    previous_partitions = set(initial_partition)
    while partition_length < half_nodes:
        new_partitions = set()
        progress_bar = tqdm(total=len(previous_partitions), 
            desc=f"Partition Gen From L={partition_length}", position=0)
        for partitions in pool.map(grow_partitions, list(previous_partitions)):
            new_partitions.extend(partitions)
            progress_bar.update(1)
        previous_partitions = new_partitions
        partition_length += 1
        np.save(f"partitions_l{partition_length}", previous_partitions)
    del previous_partitions
    # partition_length = 2
    # while partition_length <= half_nodes:
    #     partitions = np.load(f"partitions_l{partition_length}.npy")
    #     progress_bar = tqdm(total=len(partitions), 
    #         desc=f"Point Digit Conversion L={partition_length}", position=0)
    #     new_partitions = []
    #     for partition in pool.map(partition_point_to_digit, partitions):
    #         new_partitions.append(partition)
    #         progress_bar.update(1)
    #     del partitions
    #     np.save(f"partitions_l{partition_length}.npy", new_partitions)
    #     partition_length += 1
    # partition_length = 2
    # while partition_length < half_nodes:
    #     partitions = np.load(f"partitions_l{partition_length}.npy")
    #     progress_bar = tqdm(total=len(partitions), 
    #         desc=f"Digit Inversion L={partition_length}", position=0)
    #     new_partitions = []
    #     for partition in pool.map(partition_digit_inversion, partitions):
    #         new_partitions.append(partition)
    #         progress_bar.update(1)
    #     del partitions
    #     np.save(f"partitions_l{partition_length}.npy", new_partitions)
    #     partition_length += 1