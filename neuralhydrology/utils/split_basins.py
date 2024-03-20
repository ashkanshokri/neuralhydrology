"""
Module: split_basins

This module provides functions for splitting a list of names read from an input file into multiple files. 
The names are shuffled before the split to ensure randomness.

Functions:
- split_basins(input_file: str, output_directory: str, split_sizes: List[int]) -> None:
    Read names from an input file, shuffle them, and split them into files based on specified split sizes. 
    Write the split names to output files.

- k_fold_split_basins(input_file: str, output_directory: str, k: int, folds_not_equal_ok: bool = False) -> None:
    Read names from an input file, shuffle them, and perform k-fold cross-validation.
    Write the split names to output files for each fold (test and train).

Parameters:
- input_file (str): Path to the input file containing names.
- output_directory (str): Path to the directory where output files will be created.
- split_sizes (List[int]): List of integers representing the sizes of splits.

- k (int): Number of folds for the split in k_fold_split_basins.
- folds_not_equal_ok (bool): If True, allows unequal fold sizes in k_fold_split_basins. Default is False.

Returns:
None
"""

import os
import random
from typing import List


def split_basins(input_file: str, output_directory: str, split_sizes: List[int]) -> None:
    """
    Read names from an input file, shuffle them, and split them into files
    based on specified split sizes. Write the split names to output files.

    Parameters:
    - input_file (str): Path to the input file containing names.
    - output_directory (str): Path to the directory where output files will be created.
    - split_sizes (List[int]): List of integers representing the sizes of splits.

    Returns:
    None
    """
    # Read the names from the input file
    with open(input_file, 'r') as file:
        names = file.readlines()
    random.shuffle(names)

    # Check if the sum of split sizes is greater than the total number of names
    if sum(split_sizes) > len(names):
        print("Error: Sum of split sizes exceeds the total number of names.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Split the names and write to files using a counter
    start_idx = 0
    counter = 1
    for size in split_sizes:
        end_idx = start_idx + size
        output_file = os.path.join(output_directory, f'output_{size}_part{counter}.txt')

        with open(output_file, 'w') as file:
            file.writelines(names[start_idx:end_idx])

        start_idx = end_idx
        counter += 1

    # Write remaining names to a file with a name based on the number of remaining names
    if start_idx < len(names):
        remaining_file = os.path.join(output_directory, f'output_{len(names)-start_idx}_part{counter}.txt')
        with open(remaining_file, 'w') as file:
            file.writelines(names[start_idx:])


import os
import random
from typing import List


def k_fold_split_basins(input_file: str,
                        output_directory: str,
                        k: int,
                        folds_not_equal_ok: bool = False,
                        seed: int = 42) -> None:
    """
    Read names from an input file, shuffle them, and split them into files
    based on specified split sizes. Write the split names to output files.

    Parameters:
    - input_file (str): Path to the input file containing names.
    - output_directory (str): Path to the directory where output files will be created.
    - k (int): Number of folds for the split.
    - folds_not_equal_ok (bool): If True, allows unequal fold sizes. Default is False.
    - seed (int): Seed number for random shuffling. Default is 42. None will run it without seed.

    Returns:
    None
    """
    # Read the names from the input file
    with open(input_file, 'r') as file:
        names = file.readlines()

    # Set the seed for random shuffling
    if seed is not None:
        random.seed(seed)

    random.shuffle(names)

    # Calculate the split size
    split_size = len(names) // k

    # Check if len(names) is not divisible by k, and handle based on folds_not_equal_ok
    if not folds_not_equal_ok and len(names) % k != 0:
        raise ValueError(
            "Number of names is not divisible by the number of folds. Adjust k or set folds_not_equal_ok=True.")

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Split the names and write to files using a counter
    start_idx = 0
    chunks = [names[i:i + split_size] for i in range(0, len(names), split_size)]

    for fold in range(k):
        test = os.path.join(output_directory, f'fold_{fold}_test.txt')
        train = os.path.join(output_directory, f'fold_{fold}_train.txt')

        with open(test, 'w') as file:
            file.write('\n'.join(line.strip() for line in chunks[fold]))

        train_basins = [x for i, l in enumerate(chunks) for x in l if i != fold]

        with open(train, 'w') as file:
            file.write('\n'.join(line.strip() for line in train_basins))
