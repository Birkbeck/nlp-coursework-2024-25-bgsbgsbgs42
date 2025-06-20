#a
import pandas as pd

def data_processing():
    # Reading the CSV file
    df = pd.read_csv('texts/hansard40000.csv')
    
    #Check the df is not empty
    if df.empty:
        raise ValueError("Empty dataframe loaded")
    
    #Check required colums are present
    required_columns = {'Party', 'Speech', 'speech_class'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    #Check 'Party' is column is present in df 
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
    
    # Filter speech class
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
    
    vectoriser = TfidfVectorizer(stop_words='english', max_features=3000 min_df=5) # Error handling for edge cases of rare terms 
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score


def train_evaluate_models():
    try:
        # Initialise models
        randomforest = RandomForestClassifier(n_estimators=300, random_state=26)
        SVM = SVC(kernel='linear', random_state=26)
        