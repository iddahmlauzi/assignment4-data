import os
import numpy as np
from transformers import AutoTokenizer
from pathlib import Path
from cs336_data.modal_utils import VOLUME_MOUNTS, app, build_image, data_volume 

LOCAL_DATA_PATH = Path("/root/data")
INPUT_PATH = LOCAL_DATA_PATH / "exact_line_deduped_english_data"
OUTPUT_PATH = LOCAL_DATA_PATH / "tokenized_c4_filtered" 

EOS_STRING = "<|endoftext|>"


@app.function(image=build_image(),
              max_containers=25,
              volumes=VOLUME_MOUNTS,
              timeout=60 * 60 * 12)
def tokenize_file(filepath: os.PathLike):
    """
    Tokenizes the provided txt file
    This txt file already contains "<|endoftext|>" between documents"""
    OUTPUT_PATH.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    with open(filepath) as f:
        documents = f.read().split(EOS_STRING)
    
    results = []
    for doc in documents:
        tokens = tokenizer.encode(doc) + [tokenizer.eos_token_id]
        results.append(tokens)
        
    # Flatten the list of ids and convert to numpy array
    all_ids = [token_id for sublist in results for token_id in sublist]
    num_tokens = len(all_ids)
    print(f"Tokenized and encoded {filepath} into {num_tokens} tokens")
        
    ids_array = np.array(all_ids, dtype=np.uint16)
    ids_array.tofile(OUTPUT_PATH / (filepath.stem + ".bin"))
    
    return num_tokens


@app.function(image=build_image(),
              volumes=VOLUME_MOUNTS,
              timeout=60 * 60 * 12)
def main():

    OUTPUT_PATH.mkdir(exist_ok=True)
    filepaths = list(INPUT_PATH.glob("*"))
    all_token_nums = list(tokenize_file.map(filepaths)) 
    data_volume.reload()
    
    total_tokens = sum(all_token_nums)
    print(f"Total Number of tokens in dataset: {total_tokens:,}")
    
    # Concatenate the files from the folder into one .bin file
    tokenized_data_filepaths = list(OUTPUT_PATH.glob("*"))
    with open(LOCAL_DATA_PATH / "final_tokenized_english_data.bin", "ab") as f:
        for filepath in tokenized_data_filepaths:
            data = np.fromfile(filepath, dtype=np.uint16)
            data.tofile(f)
            
    print("Successfully saved the final tokenized dataset")
    

@app.local_entrypoint()
def model_main():
    main.remote()