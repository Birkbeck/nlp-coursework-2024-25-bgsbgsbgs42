#a
import pandas as pd

def data_processing():
    # Reading the CSV file
    df = pd.read_csv('texts/hansard40000.csv')
    
    #Renaming Labour (Co-op) to Labour (i)
    df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')
    
    #Keeping only the four most common parties and removing Speaker column (ii)
    top_parties = df['party'].value_counts().nlargest(4).index
    df = df[df['party'].isin(top_parties)]
    df = df[df['party'] != 'Speaker']
    
    #Keeping only the rows where speech_class is Speech (iii)
    df = df[df['speech_class'] == 'Speech']
    
    #Removing rows with speeches shorter than 1000 characters (iv)
    df = df[df['speech'].str.len() >= 1000]
    
    #Printing the dimensions of the resulting dataframe using df.shape
    print(df.shape)
    
    
#b 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def vectorise_data():
    