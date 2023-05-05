import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map
import multiprocessing as mp
import itertools
import functools
import sys

grid = [
    [0, 4, 8, 12],
    [1, 5, 9, 13],
    [2, 6, 10, 14],
    [3, 7, 11, 15]
]

partitions_dict = {}
grid_flat = list(range(0, 16))
grid_points = []
for x in range(0, 4):
    for y in range(0, 4):
        grid_points.append((x, y))
nodes = 16
half_nodes = 5
partition_length = 1

start = (0,0)
for i in range(0, half_nodes+1):
    partitions_dict[i] = []
partitions_dict[partition_length] = [[start]]

def partition_point_to_digit(points):
    return [
        grid[point[0]][point[1]] for point in points
    ]

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
    with mp.Pool(processes=8) as pool:
        previous_partitions = []
        while partition_length <= half_nodes:
            if partition_length == 1: previous_partitions = partitions_dict[1]
            filename = f"partitions_l{partition_length}.npy"
            with open(filename, "wb") as f:
                for result in tqdm(pool.imap_unordered(
                grow_partitions, previous_partitions,
                chunksize=int(len(previous_partitions)/256)+1), 
                total=len(previous_partitions), 
                desc=f"Partition Gen L={partition_length}"):
                    np.save(f, result)
            previous_partitions = np.load(filename).tolist()
            print(previous_partitions)
            partition_length += 1
        for l in range(1, half_nodes+1):
            progress_bar = tqdm(total=len(partitions_dict[l]), desc=f"Point Digit Conversion L={l}", position=0, leave=True)
            results = []
            for result in pool.imap_unordered(
            partition_point_to_digit, partitions_dict[l],
            chunksize=int(len(partitions_dict[l])/256)+1):
                progress_bar.update(1)
                results.append(result)
            partitions_dict[l] = results
            del results
    for l in range(1, half_nodes+1):
        unique_partitions = np.unique(partitions_dict[l], axis=0)
        np.save(f"partitions_l{l}.npy", unique_partitions)
        del unique_partitions
        