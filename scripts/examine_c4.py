import numpy as np
import sys
sys.path.insert(0, ".")
from transformers import AutoTokenizer
from cs336_data.filtering import identify_language



data = np.fromfile(
 "local-shared-data/tokenized_paloma_c4_100_domains_validation.bin",
 dtype=np.uint16
)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # print(tokenizer.decode(data[0:2000]))
    
    # Figure out the language threshold in C4
    bounds = np.where(data == tokenizer.eos_token_id)[0]
    bounds = np.concatenate([[0], bounds])
    
    scores = []
    num_non_english = 0
    total_docs = 0
    for i in range(len(bounds) - 1):
        total_docs += 1
        doc = tokenizer.decode(data[bounds[i]: bounds[i + 1]])
        lang, score = identify_language(doc)
        if lang == "en":
            scores.append(score)
        else:
            num_non_english += 1
         
    print(f"Total Docs: {total_docs}")   
    print(f"Non-English docs: {num_non_english}")
    percentiles = np.percentile(scores, [5, 10, 25, 50])
    print(percentiles)
            
    
    
    