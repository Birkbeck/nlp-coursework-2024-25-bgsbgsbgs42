#a
import pandas as pd
def data_processing():
    # Reading the CSV file
    df = pd.read_csv('hansard40000.csv')
    
    #Checking the df is not empty
    if df.empty:
        raise ValueError("Empty dataframe loaded")
    
    #Checking required colums are present
    required_columns = {'party', 'speech', 'speech_class'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    #Checking 'Party' is column is present in df 
    if 'party' not in df.columns:
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
    return df
    
    
#b 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Fucntion for vectorising the speeches using TF-IDF
def vectorise_data():
    # Getting cleaned data
    df = data_processing()
    
    # Ensuring no NaN values remain
    if df['party'].isna().any():
        df = df.dropna(subset=['party'])

    
    vectoriser = TfidfVectorizer(stop_words='english', max_features=3000, min_df=5) # Error handling for edge cases of rare terms 
    x = vectoriser.fit_transform(df['speech'].astype(str))
    y = df['party']
    
    if x.shape[0] == 0:
        raise ValueError("No documents could be vectorised")
    
    # Splitting into the train and test sets with stratified sampling
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=26, shuffle=True, stratify=y)
    
    
    # Validating split sizes including for sparse matrices
    if len(x_train) == 0 or len(x_test) == 0 or x_test.shape[0] == 0:
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
    # Getting cleaned data
    df = data_processing()
    
    # Ensuring no NaN values remain
    if df['party'].isna().any():
        df = df.dropna(subset=['party'])
    
    vectoriser = TfidfVectorizer(ngram_range=(1, 3),  stop_words='english', max_features=3000, min_df=5)
    x = vectoriser.fit_transform(df['speech'].astype(str))
    y = df['party']
    
    # Splitting the data
    x_train, x_test, y_train, y_test = train_test_split (x, y, test_size=0.2, random_state=26, stratify=y)
    
    # Validating split sizes including for sparse matrices
    if len(x_train) == 0 or len(x_test) == 0 or x_test.shape[0] == 0:
        raise ValueError("Empty train or test set after splitting")
    
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
import re
import spacy
import json 
from typing import List, Dict, Set
import nltk

nltk.download('stopwords')

def custom_tokeniser_political_speeches(text:str) -> list[str]:
    """
    The tokeniser must:
        Preserve political and ideological vocabulary (e.g. "freedom", "security", "tax", "immigration"), especially across the left/right spectrum ( e.g. "socialism", "neoliberalism", "one-nation conservatism", "Blairism") and value specific phrases (e.g."equality", "sovereignty", "libertarian", "populism")
        Remove stopwords and meaningless fillers (e.g. "uh", "applause") and drop filler words (e.g."erm", "hear hear", "interruption") and overly frequent non-discriminatory/ambigous terms (e.g."country", "people").
        Lemmatise words to group related forms ("running" becomes "run")
        Retain named entities (like "United States", "European Union") and Institutions (e.g."House of Lords", "First Past the Post", "Devolution", "Supreme Court",) as well as historical references (e.g. "Winter of Discontent", "New Labour", "Coalition Government")
        Filter by part-of-speech to keep nouns, verbs, adjectives 
        Include unigrams, bigrams and trigrams (like "tax cuts", "climate change") especially for political slogans (e.g."Take Back Control", "Levelling Up", "Strong and Stable") and media frames (e.g "sleaze allegations", "partygate", "Peston interview")
        Handle party names and abbreviations (e.g. Tory, LibDem, SNP..),UK political phrases(e.g. red wall, hung parliament), economic terms (e.g. quantitive easing, austerity, furlough) consitutional terms (e.g. Westminster, Whitehall, Downing Street), specific policy area vocabulary (e.g. NHS, grammar schools), Brexit related terms, UK spelling variations
        Preserve hyphenated compounds (e.g. "middle-income", "post-Brexit", "anti-Semitism", "pro-European")
        Segment for policy areas: NHS (e.g."Junior doctors", "waiting lists", "Privatisation"); education (e.g. "T-levels", "free schools", "tuition fees"); housing: (e.g."Section 21", "leasehold reform", "Right to Buy"); defence (e.g."Trident renewal", "two-carrier strategy")
        Preserve acronyms (e.g."PMQs", "FoM", "ERG", "DUP", "PFI")
        Find patterns for Political collocations (e.g "public services", "working families"), ideological phrases( e.g."social justice", "free market"), crisis terminology (e.g"cost of living", "housing crisis")
        Handle parliamentary jargon such as procedural terms (e.g. "reading", "amendment", , "whip", "backbench", "frontbench"), specific titles and roles (e.g."Right Honourable", "Prime Minister", "Chancellor", "Secretary of State", "Shadow", "Chief Whip", "Leader of the Opposition", "Backbencher", "Sinn Féin abstention") and formal address (e.g. "Mr Speaker", "Madam Deputy Speaker", "honourable member", "right hon"); debate phrases(e.g."point of order", "unparliamentary language", "division bell"); voting/legislation terms (e.g."free vote", "three-line whip", "statutory instrument", "Henry VIII powers")
   
    ingests -> text (corpus of political texts)
    outputs -> list of tokens preserving political vocabulary
   
   Steps:
    1. Text normalisation and cleaning and remove common parliamentary fillers
        text to lowercase (preserving NER)
        filler word cleaning (e.g. "erm", "uhm", "you know")
    2. UK political term normalisation: 
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
    5. N-gram extraction
    6. Final filtering and output
        combine all tokens 
        restore preserved phrases (hyphenated terms and political phrases)
        convert placeholders back to underscored token
        remove duplicates while preserving order (deduplication)
        ensure all tokens meet minimum length requirements
        return list of processed tokens
        

    """
    
    # Loading spaCy's English model
    nlp = spacy.load("en_core_web_sm")
    
    
    
    #Storing the original text 
    original_text = text
    
    # Preprocessing, begininng with potentially hyphenated terms and removing fillers
    
    text = re.sub(r'\[(?:applause|laughter|interruption|hear hear)\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:erm|uhm|uh|you know)\b', ' ', text, flags=re.IGNORECASE)
    
    text = text.lower()
    
    hyphenated_terms = [
        'anti-european', 'pro-european', 'post-brexit', 'pre-brexit',
        'middle-income', 'low-income', 'high-income', 'working-class',
        'anti-semitism', 'anti-immigrant', 'pro-business', 'cross-party',
        'long-term', 'short-term', 'medium-term', 'hard-working',
        'co-operation', 'co-ordination', 'self-determination', 'nation-building'
    ] 
    
    for term in hyphenated_terms:
        text = text.replace(term, term.replace('-', '_'))
        
    # Normalising political terms, acronyms, abbreviations, and party names
    try:
        with open('political_terms.json') as f:
            normalisations = json.load(f)

        # Converting string patterns to compiled regex
        compiled_patterns = {}
        for cat in normalisations:
            for pattern, replacement in normalisations[cat].items():
                compiled_patterns[re.compile(pattern)] = replacement
    
    except FileNotFoundError:
        print("Warning: political_terms.json not found, skipping normalisations")
        normalisations = {}
            
    # Loading key political phrases and terms from .txt files
    try:
        with open('wordsinbritishpolitics.txt', 'r') as f:
            key_words = set(line.strip().lower() for line in f)
        with open('phrasesinbritishpolitics.txt', 'r') as f:
            key_phrases = set(line.strip().lower() for line in f)
    
    except FileNotFoundError:
        print("Warning: Political vocabulary files not found")
        key_words, key_phrases = set(), set()
        
   # Defining the stopwords to remove
    political_stopwords = {
        'people', 'country', 'nation', 'government', 'party', 'member', 'members',
        'house', 'parliament', 'committee', 'minister', 'today', 'yesterday',
        'think', 'believe', 'say', 'said', 'tell', 'know', 'want', 'need',
        'time', 'year', 'years', 'way', 'make', 'take', 'come', 'go',
        'get', 'see', 'look', 'give', 'work', 'good', 'great', 'important'
    }
    
    standard_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                         'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 
                         'were', 'be', 'been', 'being', 'have', 'has', 'had', 
                         'do', 'does', 'did', 'will', 'would', 'could', 'should', 
                         'may', 'might', 'must', 'can', 'this', 'that', 'these', 
                         'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 
                         'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 
                         'her', 'its', 'our', 'their'}
    
    all_stopwords = standard_stopwords.union(political_stopwords)
    keep_pronouns = {'we', 'our', 'they', 'them', 'their', 'us'}
    all_stopwords = all_stopwords - keep_pronouns
    
    
    #Processing with spaCy
    doc_original = nlp(original_text)
    doc_processed = nlp(text)
    
    processed_tokens = []
    
    # Identifying named entities and key phrases
    named_entities = []
    for ent in doc_original.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'NORP', 'EVENT', 'LAW']:
            entity_text = ent.text.lower().replace(' ', '_')
            named_entities.append(f"NE_{entity_text}")
            
    spans = list(doc_processed.ents) + list(doc_processed.noun_chunks)
    
    # Storing the spans to protect them from being split
    protected_spans = []
    for span in spans:
        span_text = span.text.lower().replace(' ', '_')
        if any(phrase in span.text.lower() for phrase in key_phrases):
            protected_spans.append(span_text)
            
     # Passing again for token processing
    for token in doc_processed:
        # Skipping the stopwords
        if token.text in all_stopwords:
            continue
            
        # Skipping the punctuation
        if token.is_punct:
            continue
            
        # Checking if the token is part of a protected span
        in_protected_span = False
        for span in protected_spans:
            if token.text in span:
                in_protected_span = True
                break
                
        if in_protected_span:
            continue
            
        # Only keeping nouns, verbs, adjectives, and proper nouns
        if token.pos_ not in {'NOUN', 'VERB', 'ADJ', 'PROPN'}:
            continue
            
        # Lemmatise (except for proper nouns and political terms)
        if token.text not in key_words and token.pos_ != 'PROPN':
            lemma = token.lemma_
        else:
            lemma = token.text
            
        # Adding processed token
        processed_tokens.append(lemma)
    
    # Adding protected spans and entities
    processed_tokens.extend(protected_spans)
    processed_tokens.extend(named_entities)
    
    # Post-processing
    # Removing any empty tokens which remain
    processed_tokens = [token for token in processed_tokens if token.strip()]
    
    # Removing duplicates while preserving order (deduplication)
    seen = set()
    unique_tokens = []
    for token in processed_tokens:
        if token not in seen:
            seen.add(token)
            unique_tokens.append(token)

    final_tokens = unique_tokens.copy()
    
    
    
    # Adding n-grams for political collocations
    bigrams = [
        f"{final_tokens[i]}_{final_tokens[i+1]}" 
        for i in range(len(final_tokens)-1)
    ]
    
    trigrams = [
        f"{final_tokens[i]}_{final_tokens[i+1]}_{final_tokens[i+2]}" 
        for i in range(len(final_tokens)-2)
]
    
    filtered_ngrams = [ngram for ngram in bigrams + trigrams]

    # Converting protected underscores back to hyphens where appropriate
    
    for token in unique_tokens:
        if any(term.replace('-', '_') in token for term in hyphenated_terms):
            token = token.replace('_', '-')
        final_tokens.append(token)
    
    target_output = final_tokens + filtered_ngrams
    
    # Converting back underscores to hyphens for hyphenated terms
    result = []
    for token in target_output:
        # Checking if term should be a hyphenated term
        if any(term.replace('-', '_') == token for term in hyphenated_terms):
            result.append(token.replace('_', '-'))
        else:
            result.append(token)
    
    return result

def run_with_custom_tokeniser():
    # Loading and preprocessing the data
    df = pd.read_csv('hansard40000.csv')
    
    # Applying the same filters as in data_processing()
    df = df[df['speech_class'] == 'Speech']
    df = df[df['speech'].str.len() >= 1000]
    
    # Applying custom tokeniser
    print("Applying custom tokeniser to speeches...")
    df['processed_speech'] = df['speech'].apply(
        lambda x: " ".join(custom_tokeniser_political_speeches(str(x))))
    
    # Temporarily replacing the original speeches with processed ones
    original_speeches = df['speech'].copy()
    df['speech'] = df['processed_speech']
    
    # Running the original vectorisation function
    print("\nRunning vectorisation with custom tokenized speeches...")
    x, y, vectoriser = second_vectorise_class_report()
    
    
    return x, y, vectoriser

# Main function run lines
if __name__ == "__main__":
    print("Running data processing...")
    data_processing()
    
    print("\nRunning vectorisation...")
    x_train, x_test, y_train, y_test, vectoriser = vectorise_data()

    
    print("\nTraining and evaluating models...")
    results = train_evaluate_models(x_train, x_test, y_train, y_test)    
    
    print("\nRunning second vectorisation with n-grams...")
    second_vectorise_class_report()
    
    print("\nTesting custom tokenizer...")
    
    #Testing the tokensier function with a speech from the Part Two texts
    sample_speech = """We rightly took a decision to suspend face-to-face assessments following Public Health England’s guidance. We continue to keep this under review, but wherever possible, we are either doing a paper-based review or a telephone assessment, and we are automatically renewing reassessments that are due within three months by six months, and we review that on a regular basis."""
    tokens = custom_tokeniser_political_speeches(sample_speech)
    print("\nSample speech tokens:")
    print(tokens)
    x, y, vectoriser = run_with_custom_tokeniser()