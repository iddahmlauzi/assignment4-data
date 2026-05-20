import os
import json
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.parse.encoding import detect_encoding

from cs336_data.modal_utils import VOLUME_MOUNTS, app, build_image, data_volume 
from cs336_data.filtering import c4_quality_filter
from cs336_data.deduplication import exact_line_deduplication, minhash_deduplication


SHARED_DATA_PATH = Path("/shared-data")
LOCAL_DATA_PATH = Path("/root/data")
EOS_STRING = "<|endoftext|>"
MAX_REJECTED = 20 # 20 per file
MAX_PASSED = 20


@app.function(image=build_image(),
              max_containers=25,
              volumes=VOLUME_MOUNTS,
              timeout=60 * 60 * 12)
def filter_file(filepath: os.PathLike):
    """
    Given a WET file path, filters the data using C4 heuristics
    Writes output to a txt file
    """
    (LOCAL_DATA_PATH / "filtered_english_data").mkdir(exist_ok=True)
    (LOCAL_DATA_PATH / "rejected_english_data").mkdir(exist_ok=True)
    (LOCAL_DATA_PATH / "passed_english_data").mkdir(exist_ok=True)
    
    filter_counts = Counter()
    num_rejected = 0
    num_passed = 0
    with open(filepath, "rb") as f:
        filename = Path(filepath).name.split(".")[0]
        with open(LOCAL_DATA_PATH / "filtered_english_data" / f"{filename}.txt", "w") as out:
            with open( LOCAL_DATA_PATH / "rejected_english_data" / f"{filename}.jsonl", "w") as rejected:
                with open( LOCAL_DATA_PATH / "passed_english_data" / f"{filename}.jsonl", "w") as passed:
                    for record in ArchiveIterator(f, record_types=WarcRecordType.conversion):
                        html_bytes = record.reader.read()
                        enc = detect_encoding(html_bytes)
                        text = html_bytes.decode(encoding=enc, errors='ignore')
                        filter_name, cleaned_text = c4_quality_filter(text)
                        # Keep track of how filters are contibuting to the data
                        filter_counts[filter_name] += 1
                        
                        if filter_name == "passed":
                            if num_passed < MAX_PASSED:
                                passed.write(json.dumps({"text": text[:500]}) + "\n")        
                            out.write(cleaned_text + EOS_STRING) 
                            
                        else:
                            num_rejected += 1
                            if num_rejected < MAX_REJECTED:
                                rejected.write(json.dumps({"text": text[:500], "rejected_at": filter_name}) + "\n")
                    
    return filter_counts
            
    

@app.function(image=build_image(),
              volumes=VOLUME_MOUNTS,
              timeout=60 * 60 * 12)
def main():
    
    # Get the list of all (625) wet files
    print("Filtering the data....")
    (LOCAL_DATA_PATH / "filtered_english_data").mkdir(exist_ok=True)
    (LOCAL_DATA_PATH / "rejected_english_data").mkdir(exist_ok=True)
    wet_filepaths = list(Path("/shared-data/english-wet-data").glob("*.warc.wet.gz"))
    
    
    # Make output folder
    all_filter_counts = list(filter_file.map(wet_filepaths))   
    data_volume.reload()
    print("Finished filtering the data....") 
    
    total_counts = sum(all_filter_counts, Counter())
    print("Filter Counts")
    print("------------------")
    print(total_counts)
    
    # Deduplicate the dataset
    print("Deduplicating with exact_line_deduplication....")
    cleaned_text_filepaths = list((LOCAL_DATA_PATH / "filtered_english_data").glob("*"))
    output_directory = LOCAL_DATA_PATH / "exact_line_deduped_english_data"
    output_directory.mkdir(exist_ok=True)
    
    exact_line_deduplication(input_files=cleaned_text_filepaths, output_directory=output_directory)
    print("finished deduplicating the data....")
    

@app.local_entrypoint()
def model_main():
    main.remote()

