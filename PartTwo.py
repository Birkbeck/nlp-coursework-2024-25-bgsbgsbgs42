#a
import pandas as pd

def data_processing():
    # Reading the CSV file
    df = pd.read_csv('texts/hansard40000.csv')
    
    #Checking the df is not empty
    if df.empty:
        raise ValueError("Empty dataframe loaded")
    
    #Checking required colums are present
    required_columns = {'Party', 'Speech', 'speech_class'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    #Checking 'Party' is column is present in df 
    if 'Party' not in df.columns:
        raise ValueError("'Party' column missing in dataframe")


    #Renaming Labour (Co-op) to Labour (i)
    df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')
    
    
    if df['party'].nunique() < 4:
        raise ValueError("Fewer than 4 unique parties in dataset")
    
    #Keeping only the four most common parties and removing Speaker column (ii)
    top_parties = df['party'].value_counts().nlargest(4).index
    df = df[df['party'].isin(top_parties)]
    df = df[df['party'] != 'Speaker']
    
    # Filtering for speech class
    if 'speech_class' not in df.columns:
        raise ValueError("'speech_class' column missing")
    
    #Keeping only the rows where speech_class is Speech (iii)
    df = df[df['speech_class'] == 'Speech']
    
    #Removing rows with speeches shorter than 1000 characters (iv)
    df = df[df['speech'].str.len() >= 1000]
    
    #Ensuring there are speeches remaining after filtering
    if len(df) == 0:
        raise ValueError("No speeches remaining after filtering")
    
    #Printing the dimensions of the resulting dataframe using df.shape
    print(df.shape)
    
    
#b 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Fucntion for vectorising the speeches using TF-IDF
def vectorise_data():
    # Reading the CSV file
    df = pd.read_csv('texts/hansard40000.csv')
    
    vectoriser = TfidfVectorizer(stop_words='english', max_features=3000, min_df=5) # Error handling for edge cases of rare terms 
    x = vectoriser.fit_transform(df['speech'].astype(str))
    y = df['party']
    
    if x.shape[0] == 0:
        raise ValueError("No documents could be vectorized")
    
    # Splitting into the train and test sets with stratified sampling
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, 
        train_size=0.8, 
        random_state=26, 
        shuffle=True, 
        stratify=y
    )
    
    # Validating split sizes
    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("Empty train or test set after splitting")
    
    # Print out the class distributions
    print("Class distribution in the full dataset:")
    print(y.value_counts())
    print("\nClass distribution in the train set:")
    print(y_train.value_counts())
    print("\nClass distribution in the test set:")
    print(y_test.value_counts())
    
    return x_train, x_test, y_train, y_test, vectoriser

#c
#Importing the dependencies
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score


def train_evaluate_models(x_train, x_test, y_train, y_test):
    # Initialise the models
    randomforest = RandomForestClassifier(n_estimators=300, random_state=26)
    SVM = SVC(kernel='linear', random_state=26)
    
    # Training the models
    randomforest.fit(x_train, y_train)
    SVM.fit(x_train, y_train)
    
    # Obtaining the results
    results = {}
    for m, model in [('Random Forest', randomforest), ('SVM', SVM)]:
        y_pred = model.predict(x_test)
        macro_f1_score = f1_score(y_test, y_pred, average='macro')
        c_report = classification_report(y_test, y_pred)
        
        results[m] = {
                'macro_f1': macro_f1_score,
                'report': c_report
            }
        
        # Print the results 
        print(f"\n{m} Results:")
        print(f"Macro-average F1 score: {macro_f1_score:.4f}")
        print("Classification Report:")
        print(c_report)
        
        return results

#d
# Function for second vectorisor with unigrams, bigrams and trigrams
def second_vectorise_class_report():
    # Reading the CSV file
    df = pd.read_csv('texts/hansard40000.csv')
    
    vectoriser = TfidfVectorizer(ngram_range=(1, 3),  stop_words='english', max_features=3000, min_df=5)
    x = vectoriser.fit_transform(df['speech'].astype(str))
    y = df['party']
    
    # Splitting the data
    x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=26, stratify=y)
    
    randomforest = RandomForestClassifier(n_estimators=300, random_state=26)
    SVM = SVC(kernel='linear', random_state=26)
    
    # Training the models
    randomforest.fit(x_train, y_train)
    SVM.fit(x_train, y_train)
    
    # Obtaining the results
    results = {}
    for m, model in [('Random Forest', randomforest), ('SVM', SVM)]:
        y_pred = model.predict(x_test)
        macro_f1_score = f1_score(y_test, y_pred, average='macro')
        c_report = classification_report(y_test, y_pred)
        
        results[m] = {
                'macro_f1': macro_f1_score,
                'report': c_report
            }
        
        # Printing the results 
        print(f"\n{m} Results with n-grams (1-3):")
        print(f"Macro-average F1 score: {macro_f1_score:.4f}")
        print("Classification Report:")
        print(c_report)
        
    return x, y, vectoriser

#e
#Importing the dependencies 
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import normalizers
from tokenizers.normalizers import NFKC

def custom_tokeniser_political_speeches(text:str) -> list[str]:
    """
    The tokeniser must:
        Preserve political and ideological vocabulary (e.g. "freedom", "security", "tax", "immigration"), especially across the left/right spectrum ( e.g. "socialism", "neoliberalism", "one-nation conservatism", "Blairism") and value specific phrases (e.g."equality", "sovereignty", "libertarian", "populism")
        Remove stopwords and meaningless fillers (e.g. "uh", "applause") and drop filler words (e.g."erm", "hear hear", "interruption") and overly frequent non-discriminatory/ambigous terms (e.g."country", "people").
        Lemmatize words to group related forms ("running" becomes "run")
        Retain named entities (like "United States", "European Union") and Institutions (e.g."House of Lords", "First Past the Post", "Devolution", "Supreme Court",) as well as historical references (e.g. "Winter of Discontent", "New Labour", "Coalition Government")
        Filter by part-of-speech to keep nouns, verbs, adjectives 
        Include unigrams, bigrams and trigrams (like "tax cuts", "climate change") especially for political slogans (e.g."Take Back Control", "Levelling Up", "Strong and Stable") and media frames (e.g "sleaze allegations", "partygate", "Peston interview")
        Handle party names and abbreviations (e.g. Tory, LibDem, SNP..),UK political phrases(e.g. red wall, hung parliament), economic terms (e.g. quantitive easing, austerity, furlough) consitutional terms (e.g. Westminster, Whitehall, Downing Street), specific policy area vocabulary (e.g. NHS, grammar schools), Brexit related terms, UK spelling variations
        Preserve hyphenated compounds (e.g. "middle-income", "post-Brexit", "anti-Semitism", "pro-European")
        Segment for policy areas: NHS (e.g."Junior doctors", "waiting lists", "Privatisation"); education (e.g. "T-levels", "free schools", "tuition fees"); housing: (e.g."Section 21", "leasehold reform", "Right to Buy"); defence (e.g."Trident renewal", "two-carrier strategy")
        Preserve acronyms (e.g."PMQs", "FoM", "ERG", "DUP", "PFI")
        Find patterns for Political collocations (e.g"public services", "working families"), ideological phrases( e.g."social justice", "free market"), crisis terminology (e.g"cost of living", "housing crisis")
        Handle parliamentary jargon such as procedural terms (e.g. "reading", "amendment", , "whip", "backbench", "frontbench"), specific titles and roles (e.g."Right Honourable", "Prime Minister", "Chancellor", "Secretary of State", "Shadow", "Chief Whip", "Leader of the Opposition", "Backbencher", "Sinn Féin abstention") and formal address (e.g. "Mr Speaker", "Madam Deputy Speaker", "honourable member", "right hon"); debate phrases(e.g."point of order", "unparliamentary language", "division bell"); voting/legislation terms (e.g."free vote", "three-line whip", "statutory instrument", "Henry VIII powers")
   
    ingests -> text (corpus of political texts)
    outputs -> list of tokens
   
   Steps:
    1. Text normalisation and cleaning and remove common parliamentary fillers
        text to lowercase (preserving NER)
        filler word cleaning (e.g. "erm", "uhm", "you know")
    2. UK political term normalisation: 
        normalise UK spelling variations (UK -> US spelling)
        normalise party names and political terms
        acronym expansion (PMQs -> prime_ministers_questions)
    3. spaCy processing (lemmatisation, POS, NER)
        capture politicians, organisations, locations as prefixed tokens (named entitiy extraction)
        apply lemmatisation, POS tagging, and named entity recognition
        keep only NOUN, VERB, ADJ, PROPN 
        remove general + political stopwords but preserve key pronouns
    4. Political vocabulary preservation
        protect hyphenated political terms by temporarily replacing hyphens (making them single tokens)
        preserve multi-word phrases (e.g. house of commons)
        define important political phrases to preserve as single tokens
        restore preserved phrases (hyphenated terms and political phrases)
    5. N-gram extraction
    6. Final filtering and output
        combine all tokens 
        convert placeholders back to underscored token
        remove duplicates while preserving order (deduplication)
        ensure all tokens meet minimum length requirements
        return list of processed tokens
        

    """
    
    