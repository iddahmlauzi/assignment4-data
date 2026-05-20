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


# Gopher Filters
MIN_WORDS = 50
MAX_WORDS = 100_000
MIN_MEAN_WORD_LENGTH = 3
MAX_MEAN_WORD_LENGTH = 10
ELLIPSIS_LINE_THRESHOLD = 0.3
MIN_ALPHA_WORD_RATIO = 0.8


# Filtering Thresholds
LANGUAGE_THRESHOLD = 0.86
HARMFUL_THRESHOLD = 0.95

# C4 Filters
MIN_WORDS_PER_SENTENCE = 3
MIN_SENTENCES_PER_DOC = 5

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



def c4_quality_filter(text: str) -> tuple[str, str]:
    """
    Implements the filter heuristics used to make the C4 dataset
    Heuristics taken from: https://jmlr.org/papers/volume21/20-074/20-074.pdf
    
    Args:
        text (str): the input text for one document
    Returns:
       tuple[str, str]: 
            - string indicating which filter the document failed at or "passed" if the text passes 
            - The cleaned text
    """
    lower_text = text.lower()
    cleaned_text = ""
    
    # 1. Many of the scraped pages contained warnings stating that Javascript 
    #    should be enabled so we removed any line with the word Javascript
    if "javascript" in lower_text:
        return "javascript", cleaned_text
    
    # 2. Some pages had placeholder “lorem ipsum” text; 
    #    we removed any page where the phrase “lorem ipsum” appeared
    if "lorem ipsum" in lower_text:
        return "lorem ipsum", cleaned_text
    
    # 3. Some pages inadvertently contained code. Since the curly bracket “{” 
    #    appears in many programming languages (such as Javascript, widely used on the web) 
    #    but not in natural text, we removed any pages that contained a curly bracket
    if "{" in lower_text or "}" in lower_text:
        return "curly bracket", cleaned_text
    
    # 4. We used langdetect7 to filter out any pages that were not classified as English 
    #    with a probability of at least 0.99 (about 0.86 confidence when using fasttext)
    lang, lang_score = identify_language(text)
    if lang != "en" or lang_score < LANGUAGE_THRESHOLD:
        return "language", cleaned_text
    
    
    # 5. We removed any page that contained any word on the “List of Dirty, Naughty, Obscene or Otherwise Bad Words”
    #    (Here using the fasttext classifier instead)
    nsfw_pred, nsfw_score = classify_nsfw(text)
    if nsfw_pred == "nsfw" and nsfw_score >= HARMFUL_THRESHOLD:
        return "nsfw", cleaned_text
    
    toxic_pred, toxic_score = classify_toxic_speech(text)
    if toxic_pred == "toxic" and toxic_score >= HARMFUL_THRESHOLD:
        return "toxic", cleaned_text
    
    # Line level filters
    all_lines = text.split("\n")
    retained_lines = []
    for line in all_lines:
        # 6. We only retained lines that ended in a terminal punctuation mark 
        #    (i.e. a period, exclamation mark, question mark, or end quotation mark)
        if not line.endswith((".", "!", "?", '"')):
            continue
        
        # 7. Only retained lines that contained at least 3 words
        num_words = len(line.split())
        if num_words < MIN_WORDS_PER_SENTENCE:
            continue
        
        retained_lines.append(line)
        
    if len(retained_lines) < MIN_SENTENCES_PER_DOC:
        return "document too short", ""
    
    cleaned_text = "\n".join(retained_lines)
    
    return "passed", cleaned_text
    


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
    
    # Putting this here cause it exists locally so modal won't open it
    model = load_model(str(DATA_DIR / "classifiers" / "fasttext_quality_classifier.bin"))
    
    # Fasttext predict method requires input with no newline characters
    text = text.replace("\n", " ")
    
    label, prob = model.predict(text)
    prediction = label[0].split("__")[-1]
    score = prob[0]
    return prediction, score

    
    
    
    
    

    
    

