import os
import hashlib
import mmh3
import string
import unicodedata
from pathlib import Path

import nltk
nltk.download('punkt_tab')


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
                        
 
 
def normalize(text: str) -> str:
    """Normalizes the provided text to improve minhash recall"""
    # normalize whitespaces
    text = " ".join(text.split())
    
    # Remove accents
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    
    # remove punctuation
    table = str.maketrans("", "", string.punctuation)
    text = text.translate(table).lower()
    
    return text     

def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Given two texts, computes the jaccard similarity between them"""
    
    # Make the ngrams
    # Get the score
    # TODO: write this properly
    return 0
                        
def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    """
    Runs Minhash deduplication with LSH
    Args:
        input_files: a list of paths to its input files,
        num_hashes: the number of hashes to use for computing minhash signatures
        num_bands: the number of bands to use for LSH
        ngrams: the n-gram length (in words) for computing minhash signatures
        jaccard_threshold: the threshold for computing the jaccard similarity
        output_directory: the output directory to write to 
    """
    lsh_buckets = {}
    for filepath in input_files:
        with open(filepath, "r") as f:    
            text = "\n".join(f.readlines())
            
        # Normalization improves minhash recall
        text = normalize(text)
        words = text.split()
        
        # Make the signature
        signature = [float("inf")] * num_hashes
        for hash_index in range(num_hashes): 
            for i in range(len(words) - ngrams + 1):
                ngram = " ".join(words[i: i + ngrams])
                signature[hash_index] = min(signature[hash_index],  mmh3.hash(ngram, seed=hash_index))
                
        # Locality Sensitive Hashing 
        band_size = num_hashes // num_bands
        for i in range(num_bands):
            band = tuple(signature[i * band_size: (i + 1) * band_size])
            if (i , band) not in lsh_buckets:
                lsh_buckets[(i, band)] = []
            lsh_buckets[(i, band)].append(filepath)
    
    for bucket in lsh_buckets:
       filepaths = lsh_buckets[bucket]
       
       # Efficiency: Only open each file once --> and get the ngrams
       # Don't want to store a huge dict for all ngrams in all files as this is large peak memory
       all_ngrams =  []
       for filepath in filepaths:
            file_ngrams = set()
            with open(filepath) as f:
               text = "\n".join(f.readlines())
            text = normalize(text)
            words = text.split()
            for i in range(len(words) - ngrams + 1):
                ngram = " ".join(words[i: i + ngrams])
                file_ngrams.add(ngram)
            all_ngrams.append(file_ngrams)
               
        # Now that we have all the ngrams, we can cluster
       for i in range(len(all_ngrams)):
           for j in range(i + 1, len(all_ngrams)):
            file1_ngrams = all_ngrams[i]
            file2_ngrams = all_ngrams[j]
            similarity = jaccard_similarity(file1_ngrams, file2_ngrams)
            
    
                    