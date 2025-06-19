#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
import spacy
from pathlib import Path
import os


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

#import dependencies
import cmudict 

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
        
            


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    pass

#Import dependencies 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
    
def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize.
    
       ingests -> text to analyse (string)
       output -> type-token ratio of the text (float)
    """
    
    
    #Tokenize work using word tokenizer
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


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
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

