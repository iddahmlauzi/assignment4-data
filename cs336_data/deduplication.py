import os
import hashlib
from pathlib import Path


def exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    """Takes a list of paths to input files and performs exact line deduplication on them"""
    line_counts = {}
    for filepath in input_files:
        with open(filepath, "r") as f:
            for line in f:
                line_hash = hashlib.md5(line.encode()).hexdigest()
                line_counts[line_hash] = line_counts.get(line_hash, 0) + 1
                
    for filepath in input_files:
        with open(filepath, "r") as f:
            output_filepath = Path(output_directory) / Path(filepath).name
            with open(output_filepath, "w") as out:
                for line in f:
                    line_hash = hashlib.md5(line.encode()).hexdigest()
                    line_count =  line_counts.get(line_hash, 0)
                    if line_count == 1:
                        out.write(line)
                        
                        
def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    """Enter commet here"""
    pass
                    