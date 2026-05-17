import re
from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text
from pathlib import Path
from fasttext import load_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "local-shared-data"

# REGEX PATTERNS
EMAIL_PAT = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
PHONE_PAT = re.compile(r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}\b')

def extract_text(html_bytes: bytes) -> str:
    """Extracts text from a byte string containing raw HTML"""
    
    enc = detect_encoding(html_bytes)
    html_str = html_bytes.decode(encoding=enc, errors='ignore')
    return extract_plain_text(html_str)


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


def mask_emails(text: str) -> str:
    """Masks all email addresses in the provided text"""
    return EMAIL_PAT.subn("|||EMAIL_ADDRESS|||", text)

def mask_phone_numbers(text: str) -> str:
    """Masks all phone numbers in the provided text"""
    return PHONE_PAT.subn("|||PHONE_NUMBER|||", text)

    
    

