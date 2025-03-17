"""
Standardizing Placeholders
Replaces URLs, email addresses, phone numbers, money amounts, internet packages, exchange rates, transaction IDs, and dates with standardized labels (e.g., web_link, email_contact, money_amount, date_info).

Emoji Removal and Counting
Identifies and counts emojis in the text.
Removes emojis while preserving other text characters.

Stoword removal for english and burmese

Standalone Number Removal
Removes standalone numbers (both Arabic and Burmese) from the text to focus on meaningful content.

Hashtag Counting and Removal, keep text
Identifies and counts hashtags (# followed by text).
Removes the # symbol but retains the hashtag text.

Punctuation Counting
Counts occurrences of punctuation marks (e.g., !, ?, ., etc.) to analyze text structure and usage.

Text Normalization
Converts text to lowercase and removes extra spaces for consistency across the dataset.

Authour - Nu Wai Thet
"""

import pandas as pd 
import numpy as np
import re
from myTokenize import SyllableTokenizer
from myTokenize import WordTokenizer
import re
import emoji
import nltk
from nltk.corpus import stopwords
from scipy.sparse import hstack
import os

# https://github.com/ye-kyaw-thu/myStopword?tab=readme-ov-file#top-100-stopwords-with-word2vec-freq-approach
stopword_list1 = [
    "ပါ", "က", "တယ်", "ကို", "မ", "သည်", "နေ", "တာ", "တွေ", "ရ", "မှာ", "တဲ့", "များ", "တော့", "ဖြစ်", 
    "ပြီး", "နဲ့", "ရှိ", "လို့", "တို့", "လည်း", "တစ်", "ခဲ့", "ဘူး", "သူ", "ပဲ", "ဆို", "လိုက်", "မှ", "လာ", 
    "ပေး", "၏", "ကြ", "နိုင်", "သွား", "ပြော", "လား", "ရင်", "သော", "ထား", "မယ်", "တွင်", "နှင့်", "လေး", 
    "မှု", "ရေး", "လေ", "ရဲ့", "ပြီ", "လုပ်", "ချစ်", "လဲ", "ချင်", "ဒီ", "သိ", "ကောင်း", "နှစ်", "ဘာ", 
    "ပါစေ", "လို", "ဟုတ်", "အားပေး", "ဟာ", "အရမ်း", "နော်", "မြန်မာ", "ဖို့", "အတွက်", "ကြီး", "ထဲ", "ခြင်း", 
    "ခု", "ကျွန်တော်", "ပြန်", "၍", "သို့", "ယောက်", "နိုင်ငံ", "ကြည့်", "လူ", "ပေါ့", "အောင်", "သူ့", "ဦး", 
    "ရောက်", "သေး", "မည်", "မင်း", "လှ", "ကလေး", "သာ", "ရေ", "စေ", "ဘယ်", "မင်္ဂလာ", "ရာ", "သာဓု", "ဟု", 
    "မိ", "စရာ"
]

stopword_list2 = [
    "ပါ", "က", "ကို", "တယ်", "မ", "နေ", "တာ", "သည်", "မှာ", "တွေ", "ရ", "တော့", "ဖြစ်", "တဲ့", "များ", 
    "ပြီး", "လည်း", "နဲ့", "လို့", "တို့", "ရှိ", "ဆို", "ပဲ", "သူ", "ခဲ့", "ဘူး", "တစ်", "လိုက်", "မှ", "ကြ", 
    "လာ", "သွား", "ပြော", "ပေး", "၏", "လား", "ရင်", "နိုင်", "ထား", "သော", "လေ", "မယ်", "လေး", "ပြီ", "ချင်", 
    "လုပ်", "တွင်", "ချစ်", "လဲ", "နှင့်", "ဟုတ်", "သိ", "ရဲ့", "နော်", "ဘာ", "ရေး", "ဒီ", "ကောင်း", "မှု", 
    "လို", "အရမ်း", "ဟာ", "အားပေး", "ပြန်", "နှစ်", "ပါစေ", "ကြီး", "ကျွန်တော်", "ခု", "ဖို့", "ထဲ", "အတွက်", 
    "မြန်မာ", "ပေါ့", "၍", "ခြင်း", "သူ့", "ကြည့်", "လူ", "အောင်", "ယောက်", "သာ", "သေး", "နိုင်ငံ", "မင်း", 
    "သို့", "ကလေး", "ဦး", "ရောက်", "ရေ", "တတ်", "စရာ", "ရယ်", "လှ", "ဘယ်", "ရာ", "ဟု", "ကြိုက်", "မည်", 
    "တွေ့"
]
stopword_list3 = ["ရှင့်", "ဗျ", "ဗျာ","ရှ င့်" , "ရှင်","ရှ င်",] # my added list ရှင့်

burmese_stopwords =list(set(stopword_list1 + stopword_list2 + stopword_list3))

def replace_placeholders(text):
    """
    Replaces various placeholders in a given text with standardized labels.

    - Converts text to lowercase for consistent matching.
    - Replaces URLs (full links and shortened ones) with 'web_link'.
    - Replaces email addresses with 'email_contact'.
    - Replaces Telegram usernames (@username) with 'telegram_contact'.
    - Replaces Myanmar phone numbers (09xxxxxxxxx, +959xxxxxxxxx) with 'phone_contact'.
    - Replaces money amounts while keeping the associated currency keyword as 'money_amount'.
    - Replaces internet data sizes (e.g., "1GB", "950MB") with 'internet_package'.
    - Replaces internet package exchange offers (e.g., "15k=12500ks") with 'internet_package_exchange'.
    - Replaces currency exchange rates (e.g., "1USDT = 4400") with '[EXCHANGE_RATE]'.
    - Replaces transaction IDs (8-16 digit numbers) with 'transition_id'.
    - Replaces hashtags (e.g., "#example") with 'hashtag'.
    - Replaces numeric date formats (e.g., "2.2.2025", "13/01/2025") with 'date_info'.
    - Replaces Burmese numerical dates (e.g., "၁၃-၀၁-၂၀၂၅") with 'date_info'.
    - Replaces Burmese month names with date ranges (e.g., "ဇန်နဝါရီ ၁၅-၁၆") with 'date_info'.
    - Replaces English month date formats (e.g., "13 Jan 2025", "Feb 5, 2024") with 'date_info'.
    - Replaces ordinal date formats (e.g., "5th Feb 2024") with 'date_info'.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The text with placeholders replaced by standardized labels.
    """
    # Convert to lowercase for consistent matching
    text = text.lower()
    
    # Replace URLs (full links + shortened ones)
    text = re.sub(r'https?://\S+|www\.\S+|\b(bit\.ly|t\.me|viber\.me|line\.me)/\S+', 'web_link', text)

    # Replace email addresses (e.g., example@mail.com)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', 'email_contact', text)

    # Replace Telegram usernames (single @username, not @@username)
    text = re.sub(r'(?<!@)@\w+', 'telegram_contact', text)

    # Replace phone numbers (Myanmar format: 09xxxxxxxxx OR +959xxxxxxxxx, ☎️097777794, 097777794 78)
    # Replace phone numbers (Myanmar format: 09xxxxxxxxx OR +959xxxxxxxxx, also with symbols and spaces)
    text = re.sub(r'(\b☎️?\s?\+?09\d{7,9}|\b☎️?\s?\+?959\d{7,9}|\b\d{1,3}\s?\d{1,4}(\s?\d{1,4})*)\b', 'phone_contact', text)

    # Replace money amounts (only replace the number, keep the keyword)
    money_keywords = "လစာ|bonus|ရက်မှန်ကြေး|သွင်းကြေး|ကားခ|ဝန်ဆောင်ခ|kyats|mmk|kyat|ကျပ်|ks"
    money_pattern = rf'\b(\d{{1,10}})\s?({money_keywords})\b'
    text = re.sub(money_pattern, r'money_amount \2', text)  # Keep the keyword

    # Replace reversed order (keyword first, then number) → e.g., "လစာ 500000"
    money_pattern_reversed = rf'\b({money_keywords})\s*(\d{{1,10}})\b'
    text = re.sub(money_pattern_reversed, r'money_amount \1', text)  # Keep the keyword

    # Replace internet data sizes (e.g., "1gb", "2GB", "950MB")
    internet_package_pattern = r'\b(\d+)\s?(gb|mb|gb|mb)\b'
    text = re.sub(internet_package_pattern, 'internet_package', text)

    # Replace internet package exchange (e.g., "15k=12500ks")
    package_exchange_pattern = r'\b(\d+)(k|k\s*=\s*\d+)(ks|mmk|kyat|ကျပ်)\b'
    text = re.sub(package_exchange_pattern, 'internet_package_exchange', text)

    # Replace money exchange rates (e.g., "1USDT = 4400")
    exchange_rate_pattern = r'\b(\d{1,5})\s?([a-zA-Z]+)\s?=\s?(\d{1,6})\b'
    text = re.sub(exchange_rate_pattern, '[EXCHANGE_RATE] \2 = [EXCHANGE_RATE]', text)

    # Replace Transaction IDs (assuming they are numeric strings of length 8-16)
    text = re.sub(r'\b\d{8,16}\b', 'transition_id', text)

    # Replace hashtags (# followed by letters/numbers)
    text = re.sub(r'#\w+', 'hashtag', text)

    # Replace common numeric date formats (e.g., "2.2.2025", "2-2-2025", "13/01/2025")
    date_pattern = r'\b(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b'  
    text = re.sub(date_pattern, 'date_info', text)

    # Replace Burmese numerical dates (e.g., ၁၃-၀၁-၂၀၂၅ or ၁၃ရက်၁လ၂၀၂၅ခုနှစ်)
    burmese_date_pattern = r'[\u1040-\u1049]+[-/\s]*[\u1040-\u1049]+[-/\s]*[\u1040-\u1049]+'
    text = re.sub(burmese_date_pattern, 'date_info', text)

    # Replace Burmese month names & date ranges (e.g., "ဇန်နဝါရီ ၁၅-၁၆")
    burmese_months = 'ဇန်နဝါရီ|ဖေဖော်ဝါရီ|မတ်|ဧပြီ|မေ|ဇွန်|ဇူလိုင်|သြဂုတ်|စက်တင်ဘာ|အောက်တိုဘာ|နိုဝင်ဘာ|ဒီဇင်ဘာ'
    burmese_month_pattern = rf'({burmese_months})\s?[\u1040-\u1049]+[-–]\s?[\u1040-\u1049]+'  
    text = re.sub(burmese_month_pattern, 'date_info', text)

    # Replace English month names with numeric day & year (e.g., "13 Jan 2025", "Feb 5, 2024")
    english_months = 'jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december'
    english_date_pattern = rf'\b(\d{{1,2}})\s({english_months})\s(\d{{4}})\b'  
    text = re.sub(english_date_pattern, 'date_info', text)

    # Replace ordinal date formats (e.g., "5th Feb 2024")
    ordinal_date_pattern = rf'\b(\d{{1,2}}(?:st|nd|rd|th)?)\s({english_months})\s(\d{{4}})\b'  
    text = re.sub(ordinal_date_pattern, 'date_info', text)

    return text


def count_and_remove_emojis(text):
    """
    Counts the number of emojis in a given text and removes them.

    - Identifies emojis in the text using the `emoji` library.
    - Counts the total number of emojis present.
    - Removes all emojis from the text while preserving other characters.

    Args:
        text (str): The input text to be processed.

    Returns:
        tuple: A tuple containing:
            - str: The text with emojis removed.
            - int: The count of emojis found in the original text.
    """
    if not isinstance(text, str):  # Handle NaN or non-string inputs
        return "", 0  # Return empty string and 0 emoji count
    
    emoji_list = [char for char in text if emoji.is_emoji(char)]
    emoji_count = len(emoji_list)
    text = emoji.replace_emoji(text, replace='')  # Remove emojis
    return text, emoji_count


def remove_standalone_numbers(text):
    """
    Removes standalone numbers (both Arabic and Burmese) from the text.

    - Removes numbers that are not followed by letters.
    - Handles both Arabic numerals (0-9) and Burmese numerals (၀-၉).
    
    Args:
        text (str): The input text to be processed.
    
    Returns:
        str: The text with standalone numbers removed.
    """
    if not isinstance(text, str):  # Handle NaN or non-string inputs
        return ""  # Return empty string if not a valid text
    
    text = re.sub(r'\b\d+\b(?![a-zA-Z])', '', text)  # Remove numbers not followed by letters
    text = re.sub(r'[\u1040-\u1049]+(?![က-အ])', '', text)  # Remove Burmese numbers not followed by letters
    return text

def count_hashtags(text):
    """
    Counts and removes hashtags from the text.

    - Identifies words that start with `#`.
    - Counts the number of hashtags in the text.
    - Removes `#` symbols but retains the text content.
    
    Args:
        text (str): The input text to be processed.
    
    Returns:
        tuple: A tuple containing:
            - str: The text with `#` symbols removed.
            - int: The count of hashtags found.
    """
    if not isinstance(text, str):  # Handle NaN or non-string inputs
        return "", 0  # Return empty string and 0 hashtag count
    
    hashtags = re.findall(r'#\S+', text)  # Captures words that start with `#`
    hashtag_count = len(hashtags)
    text = re.sub(r'#', '', text)  # Remove the `#` symbol but keep the text
    return text, hashtag_count

def count_punctuation(text):
    """
    Counts the occurrences of each punctuation mark in the text.

    - Identifies common punctuation marks.
    - Returns a dictionary with punctuation symbols as keys and their counts as values.
    
    Args:
        text (str): The input text to be processed.
    
    Returns:
        dict: A dictionary containing punctuation marks and their respective counts.
    """
    if not isinstance(text, str):  # Handle NaN or non-string inputs
        return {}  # Return an empty dictionary if text is not valid
    
    punctuation_marks = r"[!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]"
    punctuation_count = {mark: text.count(mark) for mark in re.findall(punctuation_marks, text)}
    return punctuation_count

def normalize_text(text):
    """
    Normalizes the text by converting it to lowercase and removing extra spaces.

    - Converts all characters to lowercase.
    - Strips leading and trailing whitespace.
    
    Args:
        text (str): The input text to be normalized.
    
    Returns:
        str: The normalized text.
    """
    if not isinstance(text, str):  # Handle NaN or non-string inputs
        return ""  # Return empty string if text is not valid
    
    return text.lower().strip()  # Normalize text (lowercase, remove spaces)


# https://github.com/ye-kyaw-thu/tools/blob/master/python/clean_non_burmese.py
def separate_burmese_english(text):
    """
    Separates Burmese and English text from a given input while preserving placeholders.

    - Identifies and preserves placeholders such as [URL], [PHONE], etc.
    - Extracts Burmese words while keeping essential tone marks and structure characters.
    - Extracts English words, numbers, mentions (@username), and hashtags.

    Args:
        text (str): The input text containing a mix of Burmese and English.

    Returns:
        tuple: A tuple containing:
            - str: The extracted Burmese text.
            - str: The extracted English text.
            - list: A list of preserved placeholders.
    """
    # Preserve placeholders like [URL], [PHONE], etc.
    placeholders = re.findall(r'\[.*?\]', text)

    # Extract Burmese words while keeping tone marks and essential characters
    allowed_burmese_range = 'က-အ'
    additional_allowed_chars = 'ျြွှောါိီုူဲံ့း်္ဿဣဤဥဦဧဩဪ၌၍၏၎'
    preserve_structure_chars = ' \n'
    burmese_pattern = f'[{allowed_burmese_range}{additional_allowed_chars}{preserve_structure_chars}]+'
    burmese_text = ' '.join(re.findall(burmese_pattern, text))

    # Extract English text (keeping words, numbers, @mentions, and hashtags)
    english_text = ' '.join(re.findall(r'[a-zA-Z0-9#@]+(?:[-\w]*[a-zA-Z0-9#@]+)?', text))

    return burmese_text, english_text, placeholders


# Make sure to download the NLTK stopwords corpus if you haven't already
# import nltk
# nltk.download('stopwords')

# English stopwords (you can expand this list or use NLTK)
english_stopwords = set(stopwords.words('english'))

# Tokenize English text
def tokenize_english(text):
    """
    Tokenizes English text by splitting it into words while preserving punctuation.

    Args:
        text (str): The input English text.

    Returns:
        list: A list of tokenized words.
    """
    # Split by non-alphabetic characters (preserving words and punctuation)
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def remove_english_stopwords(text):
    """
    Removes English stopwords from the given text.

    - Tokenizes the text.
    - Filters out common stopwords.
    - Rejoins the remaining words into a sentence.

    Args:
        text (str): The input English text.

    Returns:
        str: The text without stopwords.
    """
   
    # Tokenize English text
    tokens = tokenize_english(text)
    
    # Remove stopwords
    filtered_tokens = [token for token in tokens if token.lower() not in english_stopwords]
    
    # Rejoin words into a sentence
    return ' '.join(filtered_tokens)


def remove_burmese_stopwords(text, token_method='Syllable'):
    """
    Removes Burmese stopwords from the given text.

    - Tokenizes the text based on the specified tokenization method.
    - Filters out Burmese stopwords.
    - Rejoins the remaining words into a sentence.

    Args:
        text (str): The input Burmese text.
        token_method (str): The tokenization method ('Syllable' or 'word').

    Returns:
        str: The text without stopwords.
    """

    # Tokenize text first before removing stopwords
    if token_method == 'Syllable':
        tokenizer = SyllableTokenizer()
    elif token_method == 'word':
        tokenizer = WordTokenizer(engine="CRF")  # Use "myWord", "CRF", or "LSTM"
    tokens = tokenizer.tokenize(text)

    # Remove stopwords
    filtered_words = [token for token in tokens if token not in burmese_stopwords]
    
    # Rejoin words into a sentence
    return ' '.join(filtered_words)


# Remove punctuation from a given text
def remove_english_punctuation(text):
    """
    Removes punctuation from English text while keeping letters and numbers.

    Args:
        text (str): The input English text.

    Returns:
        str: The text without punctuation.
    """

    # Remove all punctuation characters (keeping letters and numbers)
    return re.sub(r'[^\w\s]', '', text)


# Function to remove Burmese punctuation
def remove_burmese_punctuation(text):
    """
    Removes Burmese punctuation marks from the given text.

    Args:
        text (str): The input Burmese text.

    Returns:
        str: The text without Burmese punctuation.
    """
    # List of Burmese punctuation marks to remove
    burmese_punctuation = r'[၊၊။]'  # You can add more Burmese punctuations if needed
    
    # Use re.sub to replace Burmese punctuation with an empty string
    return re.sub(burmese_punctuation, '', text)


# Combine Burmese and English text after processing
def combine_text(text):
    """
    Processes and combines Burmese and English text while preserving placeholders.

    Steps:
    1. Separates Burmese and English text using `separate_burmese_english()`.
    2. Removes punctuation from both languages using `remove_burmese_punctuation()` and `remove_english_punctuation()`.
    3. Removes stopwords using `remove_burmese_stopwords()` and `remove_english_stopwords()`.
    4. Recombines the processed Burmese and English text, preserving placeholders (e.g., [URL], [PHONE]).

    Args:
        text (str): The input text containing both Burmese and English.

    Returns:
        str: The cleaned and combined text with preserved placeholders.
    """

    # Step 1: Separate Burmese and English
    burmese_text, english_text, placeholders = separate_burmese_english(text)
    
    # Step 2: Remove punctuation
    burmese_text = remove_burmese_punctuation(burmese_text,)
    english_text = remove_english_punctuation(english_text)
    
    # Step 3: Apply stopword removal
    burmese_text = remove_burmese_stopwords(burmese_text, token_method='word')  
    english_text = remove_english_stopwords(english_text)  
  
    # Step 4: Recombine while preserving placeholders
    final_text = burmese_text + ' ' + ' '.join(placeholders) + ' ' + english_text
    return final_text


def count_punctuation(text):
    """
    Counts specific punctuation marks in the given text.

    The function identifies and counts occurrences of:
    - Exclamation marks ("!", "!!", "!!!")
    - Question marks ("?", "??", "???")
    - Ellipses ("..", "...")
    - Dashes ("-", "–", "—")
    - Underscores ("_", "__", "___")

    Args:
        text (str): The input text.

    Returns:
        int: The total count of the specified punctuation marks.
    """
    punctuation_counts = {
        "exclamation_count": len(re.findall(r'!+', text)),  # Count "!", "!!", "!!!"
        "question_count": len(re.findall(r'\?+', text)),    # Count "?", "??", "???"
        "ellipsis_count": len(re.findall(r'\.{2,}', text)), # Count "..", "..."
        "dash_count": len(re.findall(r'[-–—]', text)),      # Counts "-", "–", "—"
        "underscore_count": len(re.findall(r'_+', text))    # Count "_", "__", "___"
    }
    
    # Calculate the total punctuation count
    total_punctuation_count = sum(punctuation_counts.values())
    
    return total_punctuation_count

def preprocess_text(text):
    """
    Preprocesses text by performing various cleaning steps to prepare it for analysis.

    Steps:
    1. Counts and removes emojis using `count_and_remove_emojis()`.
    2. Counts and keeps hashtags related to Burmese scams using `count_hashtags()`.
    3. Counts punctuation marks before modifying the text using `count_punctuation()`.
    4. Replaces placeholders (e.g., URLs, mentions) using `replace_placeholders()`.
    5. Removes standalone numbers that are not phone numbers or monetary amounts using `remove_standalone_numbers()`.
    6. Normalizes text by converting it to lowercase using `normalize_text()`.
    7. Combines and processes Burmese and English text while preserving placeholders using `combine_text()`.

    Args:
        text (str): The input text containing Burmese and English content.

    Returns:
        tuple: A tuple containing:
            - str: The cleaned and processed text.
            - int: The count of removed emojis.
            - int: The count of hashtags.
            - dict: A dictionary with counts of various punctuation marks.
    """
    # Step 1: Count and remove emojis
    text, emoji_count = count_and_remove_emojis(text)

    # Step 2: Count and keep hashtags (Burmese scam-related)
    text, hashtag_count = count_hashtags(text)

    # Step 3: Count punctuation before modifying text
    punctuation_counts = count_punctuation(text)

    # Step 4: Replace placeholders (URLs, mentions, etc.)
    text = replace_placeholders(text)

    # Step 5: Remove standalone numbers (not phone numbers or amounts)
    text = remove_standalone_numbers(text)

    # Step 6: Normalize text (convert to lowercase)
    text = normalize_text(text)

    # Step 7: Combine text if necessary
    text = combine_text(text)

    return text, emoji_count, hashtag_count, punctuation_counts




def load_data_and_run(save=True):
    
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, "data/labelled_money_scam_text.xlsx")
    save_path=os.path.join(script_dir,"data/preprocessed_data_wordtokenize.xlsx")
    # Load the dataset
    df = pd.read_excel(data_path)
    # Apply preprocessing and expand the results into separate columns
    # pd.Series(preprocess_text(x)) takes these multiple values and converts them into a Pandas Series (which is like a row with named values).
    df[['processed_text', 'emoji_count', 'hashtag_count', 'punctuation_counts']] = df['text'].apply(lambda x: pd.Series(preprocess_text(x)))

    # Save the preprocessed data if requested
    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the folder exists
        df.to_excel(save_path, index=False)
        print(f"Preprocessed data saved to: {save_path}")

    return df

if __name__ == "__main__":
    load_data_and_run(save=True)
    print("Data preprocessing is done sucessfully.")

