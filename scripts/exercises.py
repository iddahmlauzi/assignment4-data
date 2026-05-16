import argparse
import random
import os
import json
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.parse.encoding import detect_encoding
from pathlib import Path
from filtering import extract_text, identify_language

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "local-shared-data"
EXERCISES_DIR = PROJECT_ROOT / "data" / "exercises"


def run_extract_text_exercise():
    """ Problem (extract_text):  HTML to text conversion """
    
    # Uses the following WARC record:
    # The Digital Gamble: The Growth and Influence of Online Betting Sites
    wet_record_id = "<urn:uuid:7dacbcff-b878-4fb5-bb0a-7426496b1a45>"
    warc_record_id = "<urn:uuid:9613b297-abdc-4455-928d-010a7baffa54>"
    
    with open(DATA_DIR / "CC" / "example.warc.gz", "rb") as f:
        for record in ArchiveIterator(f):
            if record.record_id == warc_record_id:
                html_bytes = record.reader.read()
                extracted_text = extract_text(html_bytes)
                break
            
    with open(DATA_DIR / "CC" / "example.warc.wet.gz", "rb") as f:
        for record in ArchiveIterator(f):
            if record.record_id == wet_record_id:
                html_bytes = record.reader.read()
                enc = detect_encoding(html_bytes)
                wet_text = html_bytes.decode(encoding=enc)
                break

    print("Extracted Text")
    print("-" * 20)
    print(extracted_text)
    print()
    print("WET text")
    print("-" * 20)
    print(wet_text)
    print()


def run_identify_language_exercise():
    """Problem (language_identification):  Language identification"""
    all_texts = []
    with open(DATA_DIR / "CC" / "example.warc.gz", "rb") as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            html_bytes = record.reader.read()
            extracted_text = extract_text(html_bytes)
            all_texts.append(extracted_text)
    
    # Sample 20 random texts
    sampled_texts = random.sample(all_texts, k=20)
    
    result = []
    for text in sampled_texts:
        lang, score = identify_language(text)
        result.append({"lang": lang,
                       "score": score,
                       "text": text})
        
    with open(EXERCISES_DIR/"identify_language.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        
    print("Successfully ran the identify language exercise : )")
    
    
def main(args):   
    if args.extract_text:
        run_extract_text_exercise()
    
    if args.identify_lang:
        run_identify_language_exercise()


if __name__ == "__main__":
    random.seed(42)
    os.makedirs(EXERCISES_DIR, exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract_text", action="store_true")
    parser.add_argument("--identify_lang", action="store_true")
    args = parser.parse_args()
    
    main(args)
    

