import re
from typing import Any
from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text
from pathlib import Path
from fasttext import load_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "local-shared-data"

# REGEX PATTERNS
EMAIL_PAT = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
PHONE_PAT = re.compile(r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}\b')
IP_PAT = re.compile(r'(((?!25?[6-9])[12]\d|[1-9])?\d\.?\b){4}\b')

# Models
LANGUAGE_MODEL = load_model(str(DATA_DIR / "classifiers" / "lid.176.bin"))
NSFW_MODEL = load_model(str(DATA_DIR / "classifiers" / "dolma_fasttext_nsfw_jigsaw_model.bin"))
TOXIC_SPEECH_MODEL = load_model(str(DATA_DIR / "classifiers" / "dolma_fasttext_hatespeech_jigsaw_model.bin"))

def extract_text(html_bytes: bytes) -> str:
    """Extracts text from a byte string containing raw HTML"""
    
    enc = detect_encoding(html_bytes)
    html_str = html_bytes.decode(encoding=enc, errors='ignore')
    return extract_plain_text(html_str)


def identify_language(text) -> tuple[str, float]:
    """Identifies the main language in the given Unicode String"""
    
    # Fasttext predict method requires input with no newline characters
    text = text.replace("\n", " ")
    
    # The label is formatted as: '__label__ja' --> extract last part
    label, prob = LANGUAGE_MODEL.predict(text)
    lang = label[0].split("__")[-1]
    score = prob[0]
    
    return lang, score


def mask_emails(text: str) -> str:
    """Masks all email addresses in the provided text"""
    return EMAIL_PAT.subn("|||EMAIL_ADDRESS|||", text)

def mask_phone_numbers(text: str) -> str:
    """Masks all phone numbers in the provided text"""
    return PHONE_PAT.subn("|||PHONE_NUMBER|||", text)

def mask_ip_addresses(text: str) -> str:
    """Masks all IP addresses in the provided text"""
    return IP_PAT.subn("|||IP_ADDRESS|||", text)
    
def classify_nsfw(text: str) -> tuple[Any, float]:
    """
    Labels the given string as containing NSFW content or not.
    Returns a pair containing both the label and a confidence score
    """
    
    # Fasttext predict method requires input with no newline characters
    text = text.replace("\n", " ")
    label, prob = NSFW_MODEL.predict(text)
    prediction = label[0].split("__")[-1]
    score = prob[0]
    return prediction, score

def classify_toxic_speech(text: str) -> tuple[Any, float]:
    """
    Labels the given string as consisting of toxic speech or not
    Returns a pair containing both the label and a confidence score
    """
    # Fasttext predict method requires input with no newline characters
    text = text.replace("\n", " ")
    label, prob = TOXIC_SPEECH_MODEL.predict(text)
    prediction = label[0].split("__")[-1]
    score = prob[0]
    return prediction, score
    
    

    
    

