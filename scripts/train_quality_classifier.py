import gzip
import fasttext
import random
import string
import subprocess
from pathlib import Path
from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.modal_utils import VOLUME_MOUNTS, app, build_image

from cs336_data.filtering import (
    extract_text, 
    is_high_quality
)

# Number of processes to use
NUM_WIKIPEDIA_URLS = 25_000
NUM_DOWNLOAD_CHUNKS = 25 

SHARED_DATA_PATH = Path("/shared-data")
LOCAL_DATA_PATH = Path("/root/data")


def subsample_urls():
    """Subsample urls from Wikepedia"""
    
    all_urls = []
    with gzip.open(SHARED_DATA_PATH / "wiki" /"enwiki-20260501-extracted_urls.txt.gz", "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            all_urls.append(line)
            
    # Sample k URLs
    sampled_urls = random.sample(all_urls, k=NUM_WIKIPEDIA_URLS)
    
    with open(LOCAL_DATA_PATH / "subsampled_positive_urls.txt", "w") as f:
        f.writelines(url + "\n" for url in sampled_urls)
    
    print(f"Successfully sampled {NUM_WIKIPEDIA_URLS} Wikepedia URLS")

    
@app.function(image=build_image(),
              volumes=VOLUME_MOUNTS,
              timeout=60 * 60 * 12)
def download_urls(index, urls: list):
    """Downloads the provided urls"""
    
    # Write the urls to a temp file
    chunk_url_path = LOCAL_DATA_PATH / f"chunk_{index}_urls.txt"
  
    
    with open(chunk_url_path, "w") as f:
        f.writelines(url + "\n" for url in urls)

        
    # Download URL files
    (LOCAL_DATA_PATH / "warc_chunks").mkdir(exist_ok=True)
    chunk_warc_path = LOCAL_DATA_PATH / "warc_chunks" / f"chunk_{index}"
    subprocess.run(["wget", 
                    "--tries=3"
                    "--timeout=15", 
                    "--quiet",
                    "-i", str(chunk_url_path), 
                    f"--warc-file={chunk_warc_path}", 
                    "-O", "/dev/null"])
    
    # Clean up temp files 
    chunk_url_path.unlink()
    print(f"Successfully downloaded urls for chunk {index} ")
    
    
@app.function(image=build_image(),
              volumes=VOLUME_MOUNTS,
              timeout=60 * 60 * 12)
def prepare_positive_examples(index):
    """Prepares the positive examples for the provided chunk index"""
    
    all_texts = []
    with open(LOCAL_DATA_PATH / "warc_chunks" / f"chunk_{index}.warc.warc.gz", "rb") as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            try:
                html_bytes = record.reader.read()
            except Exception:
                continue
            text = extract_text(html_bytes)
            if is_high_quality(text):
                all_texts.append(text)
    
    (LOCAL_DATA_PATH / "positive_examples").mkdir(exist_ok=True)
    with open(LOCAL_DATA_PATH / "positive_examples" / f"chunk_{index}.txt", "w") as f:
        f.writelines(text + "\n" for text in all_texts)
        

def prepare_negative_examples():
    """Prepares the negative examples """     
    
    all_texts = []
    with open(SHARED_DATA_PATH / "CC" / "example.warc.gz", "rb") as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            html_bytes = record.reader.read()
            text = extract_text(html_bytes)
            if is_high_quality(text):
                all_texts.append(text)
                
    with open(LOCAL_DATA_PATH / "negative_examples.txt", "w") as f:
        f.writelines(text + "\n" for text in all_texts)
        
        
def format_line_for_fasttext(line: str):
    """formats the provide line to train a fasttext classifier"""
    
    # Remove newline characters --> fasttext does not handle these well
    line = line.strip().replace("\n", " ")
    
    # Remove punctuation --> Helps with wuality
    table = str.maketrans("", "", string.punctuation)
    line = line.translate(table).lower()
    
    return line
    

def format_training_data_for_fastext():
    """
    After preparing the positive and negative examples, we need to store these 
    in the appropriate format to train the fasttext classifier
    """
    
    with open(LOCAL_DATA_PATH / "fasttext_training.txt", "w") as out:
        for i in range(NUM_DOWNLOAD_CHUNKS):
            with open(LOCAL_DATA_PATH / "positive_examples" / f"chunk_{i}.txt", "r") as f:
                for line in f:
                    line = format_line_for_fasttext(line)
                    out.write("__label__positive " + line + "\n")
                    
        
        with open(LOCAL_DATA_PATH / "negative_examples.txt", "r") as f:
            for line in f:
                line = format_line_for_fasttext(line)
                out.write("__label__negative " + line + "\n")
                
    print("Successfulyl formateed the training data for the fasttext classifier")
    

def train_classifier():
    """Trains the fasttext classifier"""
    model = fasttext.train_supervised(input=str(LOCAL_DATA_PATH / "fasttext_training.txt"))
    model.save_model(str(LOCAL_DATA_PATH / "fasttext_quality_classifier.bin"))    
    
    print("Successfully finished training the quality classifier!!!")        
    

@app.function(image=build_image(),
              volumes=VOLUME_MOUNTS,
              timeout=60 * 60 * 12)
def main(subsample: bool, 
         run_download: bool,
         prepare_train_data: bool,
         train: bool):
    random.seed(42)
    if subsample:
        subsample_urls()
        
    if run_download:
        with open(LOCAL_DATA_PATH / "subsampled_positive_urls.txt", "r") as f:
            urls = [line.strip() for line in f.readlines()]
            chunk_size = len(urls) // NUM_DOWNLOAD_CHUNKS
            chunks = []
            for i in range(0, len(urls), chunk_size):
                chunks.append(urls[i: i + chunk_size])
            list(download_urls.starmap(enumerate(chunks)))
            
    if prepare_train_data:
        list(prepare_positive_examples.map(range(NUM_DOWNLOAD_CHUNKS)))
        prepare_negative_examples()
        format_training_data_for_fastext()
        
    if train:
        train_classifier()
            

@app.local_entrypoint()
def model_main(subsample: bool = False,
               run_download: bool = False,
               prepare_train_data: bool=False,
               train: bool=False):
    main.remote(subsample=subsample, 
                run_download=run_download, 
                prepare_train_data=prepare_train_data,
                train=train)
        
    
        
    
    
