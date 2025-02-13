import os
import pickle
import pandas as pd

def load_pickles_from_directory(directory):
    data_list = [] 
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "rb") as f:
                data = pickle.load(f)  
                data_list.append(data)
    
    return pd.DataFrame(data_list)


directory_path = "testing_performance"
df = load_pickles_from_directory(directory_path) 
# df.iloc[0]['performance']['test_accuracy'] - can index into value using this indexing (this is for first row)