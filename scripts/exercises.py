import argparse
import random
import os
import json
from tqdm import tqdm
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.parse.encoding import detect_encoding
from pathlib import Path
from cs336_data.filtering import (
    extract_text, 
    identify_language,
    mask_emails,
    mask_phone_numbers,
    mask_ip_addresses,
    classify_toxic_speech,
    classify_nsfw,
    gopher_quality_filter
)

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
        for record in tqdm(ArchiveIterator(f, record_types=WarcRecordType.response)):
            html_bytes = record.reader.read()
            extracted_text = extract_text(html_bytes)
            all_texts.append(extracted_text)
    
    # Sample 20 random texts
    sampled_texts = random.sample(all_texts, k=200)
    
    result = []
    for text in sampled_texts:
        lang, score = identify_language(text)
        result.append({"lang": lang,
                       "score": score,
                       "text": text})
        
    with open(EXERCISES_DIR/"identify_language.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        
    print("Successfully ran the identify language exercise : )")
    
    
def run_mask_pii_exercise():
    """Problem (mask_pii):  Personally identifiable information"""
    all_texts = []
    with open(DATA_DIR / "CC" / "example.warc.gz", "rb") as f:
        for record in tqdm(ArchiveIterator(f, record_types=WarcRecordType.response)):
            html_bytes = record.reader.read()
            extracted_text = extract_text(html_bytes)
            
            # Masking of PII
            masked_text, email_n = mask_emails(extracted_text)
            masked_text, phone_n = mask_phone_numbers(masked_text)
            masked_text, ip_n = mask_ip_addresses(masked_text)
            
            if email_n + phone_n + ip_n > 0:
                all_texts.append({"original_text": extracted_text,
                                  "masked_text": masked_text,
                                  "email_n": email_n,
                                  "phone_n": phone_n,
                                  "ip_n": ip_n})
                
    # Sample 20 random texts
    sampled_texts = random.sample(all_texts, k=20)
        
    with open(EXERCISES_DIR/"mask_pii.json", "w") as f:
        json.dump(sampled_texts, f, indent=2, ensure_ascii=False)
        
    print("Successfully ran the mask pii exercise : )")
    
    
def run_harmful_content_exercise():
    """Problem (harmful_content):  Harmful content"""
    all_texts = []
    with open(DATA_DIR / "CC" / "example.warc.gz", "rb") as f:
        for record in tqdm(ArchiveIterator(f, record_types=WarcRecordType.response)):
            html_bytes = record.reader.read()
            extracted_text = extract_text(html_bytes)
            lang, _ = identify_language(extracted_text)
            if lang == "en":
                all_texts.append(extracted_text)
            
    # Sample 20 random texts
    sampled_texts = random.sample(all_texts, k=100)
    
    result = []
    for text in sampled_texts:
        nsfw_pred, nsfw_score = classify_nsfw(text)
        toxic_pred, toxic_score = classify_toxic_speech(text)
        result.append({"text": text,
                       "nsfw_pred": nsfw_pred,
                       "nsfw_score": nsfw_score,
                       "toxic_pred": toxic_pred,
                       "toxic_score": toxic_score
                      })
        
    with open(EXERCISES_DIR/"harmful_content.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        
    print("Successfully ran the harmful content exercise : )")
    

def run_gopher_quality_filters_exercise():
    """Problem (gopher_quality_filters)"""
    all_texts = []
    with open(DATA_DIR / "CC" / "example.warc.gz", "rb") as f:
        for record in tqdm(ArchiveIterator(f, record_types=WarcRecordType.response)):
            html_bytes = record.reader.read()
            extracted_text = extract_text(html_bytes)
            lang, _ = identify_language(extracted_text)
            if lang == "en":
                all_texts.append(extracted_text)
                
    # Sample 20 random texts
    sampled_texts = random.sample(all_texts, k=20)
    
    result = []
    for text in sampled_texts:
        is_good_quality = gopher_quality_filter(text)
        result.append({
            "text": text,
            "is_good_quality": is_good_quality
        })
        
    with open(EXERCISES_DIR / "gopher_quality_filters.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        
    print("Successfully ran the Gopher Quality Filters Exercise")
            
    
    
def main(args):   
    if args.extract_text:
        run_extract_text_exercise()
    
    if args.identify_lang:
        run_identify_language_exercise()
        
    if args.mask_pii:
        run_mask_pii_exercise()
        
    if args.harmful_content:
        run_harmful_content_exercise()
        
    if args.gopher:
        run_gopher_quality_filters_exercise()


if __name__ == "__main__":
    os.makedirs(EXERCISES_DIR, exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract_text", action="store_true")
    parser.add_argument("--identify_lang", action="store_true")
    parser.add_argument("--mask_pii", action="store_true")
    parser.add_argument("--harmful_content", action="store_true")
    parser.add_argument("--gopher", action="store_true")
    args = parser.parse_args()
    
    main(args)
    

