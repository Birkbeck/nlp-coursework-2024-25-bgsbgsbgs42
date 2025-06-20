#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
import spacy
from pathlib import Path
import os


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

#Import dependencies 
from nltk.corpus import cmudict

d = cmudict.dict() 

# Download required NLTK data
nltk.download('punkt')
nltk.download('cmudict')

def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    # Tokenize sentences and words
    sentences = nltk.sent_tokenize(text)
    words = [word.lower() for word in nltk.word_tokenize(text) if word.isalpha()]
    
    # Calculate basic counts
    total_sentences = len(sentences)
    total_words = len(words)
    total_syllables = 0
    
    # Calculate total syllables using CMU dict and fallback method
    for word in words:
        # Try to get syllable count from CMU dictionary first
        if word in d:
            # Get syllable count from CMU dictionary (count stress markers) - first pronunciation variant
            pronunciation = d[word][0]
            syllables = sum(1 for phoneme in pronunciation if phoneme[-1].isdigit())
            total_syllables += syllables
        else:
            # Fallback syllable counting for words not in CMU dict
            vowels = "aeiouy"
            syllable_count = 0
            prev_char_was_vowel = False
            
        # Count vowel clusters
            for char in word:
                if char in vowels:
                    if not prev_char_was_vowel:
                        syllable_count += 1
                    prev_char_was_vowel = True
                else:
                    prev_char_was_vowel = False
            
            # Adjust for silent e at end
            if word.endswith('e') and syllable_count > 1:
                syllable_count -= 1
            
            # Ensure at least one syllable
            syllable_count = max(1, syllable_count)
            total_syllables += syllable_count
            
    # Calculate Flesch-Kincaid Grade Level
    if total_sentences == 0 or total_words == 0:
        return 0.0  # Avoid division by zero errors

    avg_words_per_sentence = total_words / total_sentences
    avg_syllables_per_word = total_syllables / total_words
    
    fk_score = (0.39 * avg_words_per_sentence) + (11.8 * avg_syllables_per_word) - 15.59
    return fk_score

def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    pass

#Import pandas 
import pandas as pd

def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year
    
    ingests -> file path
    output -> DataFrame containing novel information with columns: text, title, author, year

    """
    
    #Initialise array to store data
    novelData = []
    
    #Check file path and return error if not found
    if not path.exists():
        raise FileNotFoundError(f"The required file path {path} does not exist.")
    
    #For each text (.txt file) in file path
    for file_path in path.glob('*.txt'):
        try:
            # Read filename components (Title-Author-Year.txt) for content sought
            
            filename = file_path.stem  # Get filename without extension
            parts = filename.split('-')
            
            # Ensure we have at least title, author, and year, so that the dataframe is complete 
            if len(parts) < 3:
                print(f"Skipping file {filename} - incorrect naming format")
                continue
            
            # Take the metadata from the filename
            title = '-'.join(parts[:-2]).strip()  # Handle titles with hyphens
            author = parts[-2].strip()
            year = parts[-1].strip()
            
            #Check if the year is a number (i.e. in a numerical format)
            if not year.isdigit():
                print(f"Skipping file {filename} - year is not numeric")
                continue
            year = int(year)
            
            # Read the text content
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                
            # Add to the data collection
            novelData.append({
                'text': text,
                'title': title,
                'author': author,
                'year': year
            })
            
        except Exception as e:
            print (f"Error processing file {file_path}: {str(e)}")
            continue
            
    # Create the DataFrame and sort by year
    if not novelData:
        raise ValueError("No valid novel files found in the directory.")
    
    df = pd.DataFrame(novelData)
    df = df.sort_values('year').reset_index(drop=True)
    
    return df
        
            
import pickle

def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file
    
    ingests -> DataFrame, storepath, name of the pickle fite that will be output
    outputs -> pickle file of the resulting DataFrame with an added 'parsed' column containing objects in the spaCy docs"""
    # Create the directory if it doesn't exist
    store_path.mkdir(parents=True, exist_ok=True)
    pickle_path = store_path / out_name
    
    # Process texts with spaCy, with special attention to handling potential long texts
    def processing (text):
        # Check if text exceeds spaCy's max length
        max_length = nlp.max_length
        if len(text) > max_length:
            # Process in chunks if text is too long
            docs = []
            for chunk in [text[i:i+max_length] for i in range(0, len(text), max_length)]:
                docs.append(nlp(chunk))
            # Combine the docs which could lose some cross-chunk context! 
            return spacy.tokens.Doc.from_docs(docs)
        else:
            return nlp(text)
        
     # Processing each text in the DataFrame
    parsed_df  = df['text'].apply(processing)
   
    # Serialise the DataFrame to pickle
    with open(pickle_path, 'wb') as f:
        pickle.dump(parsed_df, f)

    # Return the processed DataFrame
    return parsed_df

#Import dependencies 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
    
def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize.
    
       ingests -> text to analyse (string)
       output -> type-token ratio of the text (float)
    """
    #Tokenise the work using word tokenizer
    tokens = word_tokenize(text)
    
    #Convert to LC and filter out punctuation
    words = [word.lower() for word in tokens if word.isalpha()]

    #Calculate type- token ratio 
    if not words:  # Avoid division by zero error
        return 0.0
    
    num_types = len(set(words))  # Unique words (types)
    num_tokens = len(words)      # Total words (tokens)
    
    ttratio = (num_types / num_tokens) * 100
    
    return ttratio


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results

#Import the required dependencies
import math
from collections import defaultdict
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.tokenize import word_tokenize

def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list.
    
        ingests-> parsed document (spacy.tokens.Doc) and the target verb (str)
        outputs -> list of subjects/syntactic objects and pmi score, ordered by the pmi score (list)
    """
    # Extract all the subject-verb pairs
    pairs = []
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_ == target_verb:
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    pairs.append((child.text.lower(), token.lemma_.lower()))
    
    # If no pairs are found, it will return empty list
    if not pairs:
        return []
    
    # Calculate frequencies
    word_freq = defaultdict(int)
    pair_freq = defaultdict(int)
    total_pairs = len(pairs)
    
    for subject, verb in pairs:
        word_freq[subject] += 1
        word_freq[verb] += 1
        pair_freq[(subject, verb)] += 1
    
    # Calculate the PMI scores 
    pmi_scores = []
    for (subject, verb), freq in pair_freq.items():
        # USe joint probability
        p_pair = freq / total_pairs
        
        # Use marginal probabilities
        p_subject = word_freq[subject] / total_pairs
        p_verb = word_freq[verb] / total_pairs
        
        # The PMI calculation with smoothing
        if p_pair > 0 and p_subject > 0 and p_verb > 0:
            pmi = math.log2(p_pair / (p_subject * p_verb))
            pmi_scores.append((subject, pmi))
    
    # Return top 10 by their PMI score
    return sorted(pmi_scores, key=lambda x: x[1], reverse=True)[:10]


# Import the dependencies for ii and iii
from collections import Counter

def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list.
    
    ingests -> parsed document (spacy.tokens.Doc), base form of the verb (str)
    outputs -> list of, e.g. subject, count, tuples sorted by count descending (list)
    
    """
    subjects = []
    
    for token in doc:
        # Check if token is the target verb, which works for any tense
        if token.pos_ == "VERB" and token.lemma_ == verb:
            # Find all the subjects of this verb
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):  # Search for both active and passive subjects
                    subjects.append(child.text.lower())
    
    # Return top 10 most common subjects
    return Counter(subjects).most_common(10)



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples.
    
    ingests -> doc (spacy.tokens.Doc) of the  parsed document (str)
    outputs -> list of (adjective, count) tuples sorted by count descending (list)  
    
    """
    pass



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    #path = Path.cwd() / "p1-texts" / "novels"
    #print(path)
    #df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    #print(df.head())
    #nltk.download("cmudict")
    #parse(df)
    #print(df.head())
    #print(get_ttrs(df))
    #print(get_fks(df))
    #df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # print(adjective_counts(df))
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """

