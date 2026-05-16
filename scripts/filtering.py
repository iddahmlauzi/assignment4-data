import argparse
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text
from pathlib import Path
from fasttext import load_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "local-shared-data"

def extract_text(html_bytes: bytes) -> str:
    """Extracts text from a byte string containing raw HTML"""
    
    enc = detect_encoding(html_bytes)
    html_str = html_bytes.decode(encoding=enc)
    return extract_plain_text(html_str)

def extract_text_for_warc_file(target_id: str) -> str:
    """Extracts the text for a single WARC file"""
    
    with open(DATA_DIR / "CC" / "example.warc.gz", "rb") as f:
        for record in ArchiveIterator(f):
            if record.record_id == target_id:
                html_bytes = record.reader.read()
                extracted_text = extract_text(html_bytes)
                break
                
    return extracted_text


def read_text_from_wet_file(target_id: str) -> str:
    """Reads the text from a single WET file"""
                 
    with open(DATA_DIR / "CC" / "example.warc.wet.gz", "rb") as f:
        for record in ArchiveIterator(f):
            if record.record_id == target_id:
                html_bytes = record.reader.read()
                enc = detect_encoding(html_bytes)
                wet_text = html_bytes.decode(encoding=enc)
                break
                
    return wet_text


def identify_language(text) -> tuple[str, float]:
    """Identifies the main language in the given Unicode String"""
    model_path = str(DATA_DIR / "classifiers" / "lid.176.bin")
    model = load_model(model_path)
    
    # Fasttext predict method requires input with no newline characters
    text = text.replace("\n", " ")
    
    # The label is formatted as: '__label__ja' --> extract last part
    label, prob = model.predict(text)
    lang = label[0].split("__")[-1]
    score = prob[0]
    
    return lang, score
    
    
def main(args):   
    # Problem (extract_text):  HTML to text conversion 
    # The Digital Gamble: The Growth and Influence of Online Betting Sites - afrique
    if args.extract_text:
        wet_record_id = "<urn:uuid:7dacbcff-b878-4fb5-bb0a-7426496b1a45>"
        warc_record_id = "<urn:uuid:9613b297-abdc-4455-928d-010a7baffa54>"
        extracted_text = extract_text_for_warc_file(target_id=warc_record_id)
        wet_text = read_text_from_wet_file(target_id=wet_record_id)
        
        print("Extracted Text")
        print("-" * 20)
        print(extracted_text)
        print()
        print("WET text")
        print("-" * 20)
        print(wet_text)
        print()
    
    # Problem (language_identification):  Language identification
    identify_language("こんにちは、私はアメリカで勉強している学生です")

    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--extract_text", action="store_true")
    
    args = parser.parse_args()
    main(args)
    

