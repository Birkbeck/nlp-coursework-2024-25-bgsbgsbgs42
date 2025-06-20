#a
import pandas as pd

def data_processing():
    
    # Read the CSV file
    df = pd.read_csv('texts/hansard40000.csv')
    
    #Rename Labour (Co-op) to Labour
    df['party'] = df['party'].replace('Labour (Co-op)', 'Labour')
    
    