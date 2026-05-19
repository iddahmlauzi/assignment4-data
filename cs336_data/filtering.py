import re
import sys
from typing import Any
from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text
from pathlib import Path
from fasttext import load_model
from cs336_data.common import get_shared_assets_path

import nltk
nltk.download('punkt_tab')

DATA_DIR = get_shared_assets_path()

# REGEX PATTERNS
EMAIL_PAT = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
PHONE_PAT = re.compile(r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}\b')
IP_PAT = re.compile(r'(((?!25?[6-9])[12]\d|[1-9])?\d\.?\b){4}\b')

# Models
LANGUAGE_MODEL = load_model(str(DATA_DIR / "classifiers" / "lid.176.bin"))
NSFW_MODEL = load_model(str(DATA_DIR / "classifiers" / "dolma_fasttext_nsfw_jigsaw_model.bin"))
TOXIC_SPEECH_MODEL = load_model(str(DATA_DIR / "classifiers" / "dolma_fasttext_hatespeech_jigsaw_model.bin"))
#QUALITY_CLASSIFIER_MODEL = load_model(str(DATA_DIR / "classifiers" / "fasttext_quality_classifier.bin"))
QUALITY_CLASSIFIER_MODEL = None

# Gopher Filters
MIN_WORDS = 50
MAX_WORDS = 100_000
MIN_MEAN_WORD_LENGTH = 3
MAX_MEAN_WORD_LENGTH = 10
ELLIPSIS_LINE_THRESHOLD = 0.3
MIN_ALPHA_WORD_RATIO = 0.8


# Filtering Thresholds
LANGUAGE_THRESHOLD = 0.5
HARMFUL_THRESHOLD = 0.95

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

def gopher_quality_filter(text: str) -> bool:
    """
    Implements a subset of the Gopher quality filters
    Returns a boolean indicating whether the text passes the filters
    """
    words = nltk.tokenize.word_tokenize(text)
    
    # Remove documents that:
    # 1. Contain less than 50 or more than 100,000 words.
    if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
        return False
    
    # 2. Have a mean word length outside the range of 3 to 10 characters.
    mean_word_length = sum(len(w) for w in words) / len(words)
    if mean_word_length < MIN_MEAN_WORD_LENGTH or mean_word_length > MAX_MEAN_WORD_LENGTH:
        return False
    
    # 3. Have more than 30% of lines ending with an ellipsis (“...”).
    lines = text.split("\n")
    ellipses_pctg = sum(1 for line in lines if line.endswith("...")) / len(lines)
    if ellipses_pctg > ELLIPSIS_LINE_THRESHOLD:
        return False
    
    # 4. Contain less than 80% of words with at least one alphabetic character.
    alpha_pctg = sum(any(c.isalpha() for c in word) for word in words) / len(words)
    if alpha_pctg < MIN_ALPHA_WORD_RATIO:
        return False
    
    return True


def is_high_quality(text: str) -> bool:
    """Given a text, applies filters and returns a bool indicating whether the text is high quality"""
    
    # Start with heuristic filter since this doesn't need any model calls
    if not gopher_quality_filter(text):
        return False
    
    # Only want english texts 
    lang, lang_score = identify_language(text)
    if lang != "en" or lang_score < LANGUAGE_THRESHOLD:
        return False
    
    # Remove harmful texts
    nsfw_pred, nsfw_score = classify_nsfw(text)
    if nsfw_pred == "nsfw" and nsfw_score >= HARMFUL_THRESHOLD:
        return False
    
    toxic_pred, toxic_score = classify_toxic_speech(text)
    if toxic_pred == "toxic" and toxic_score >= HARMFUL_THRESHOLD:
        return False
    
    return True

def classify_quality(text: str) -> tuple[Any, float]:
    """Uses the trained quality classifier to classify the quality of the text"""
    
    label, prob = QUALITY_CLASSIFIER_MODEL.predict(text)
    prediction = label[0].split("__")[-1]
    score = prob[0]
    return prediction, score
    
    
    
    
    

    
    

