import pandas as pd
import numpy as np
from collections import namedtuple
import torch
import torch.nn.functional as F

from sklearn.preprocessing import OneHotEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
InfoData = namedtuple("InfoData", ["input_dim", "output_dim", "is_categorical"])

# TODO: I couldn't find the CPU or Forest datasets from the Fastfood paper.

def train_test_split(x, y):
    '''
    Split data into a train (90%) and test (10%) group
    '''
    assert x.shape[0] == y.shape[0]
    split = int(0.9 * x.shape[0])
    random_idx = [torch.randperm(x.shape[0])]
    
    return x[:split], y[:split], x[split:], y[split:]

def load_iris():
    '''
    n = 149
    regression features -- 4 columns of data
    categorical target -- 3 classes
    '''
    data = pd.read_csv("./data/Iris/iris.data")
    x = data.iloc[:, :-1].to_numpy().astype(float)
    x = torch.from_numpy(x).to(device)
    # pad x to the next power of 2
    pad_size = int(2**np.ceil(np.log2(x.size(1)))) - x.size(1)
    x = F.pad(x, (0, pad_size))

    y = data.iloc[:, -1].to_numpy().astype(str)
    encoder = OneHotEncoder(sparse_output=False) # sparse_output=False ensures data is returned as a np.array
    y = encoder.fit_transform(y.reshape(-1,1))
    y = torch.from_numpy(y).to(device)

    xtrain, ytrain, xtest, ytest = train_test_split(x, y)
    info = InfoData(4, 3, True)
    return xtrain, ytrain, xtest, ytest, info


def load_animal_center():
    '''
    n = 170,588
    categorical features -- 8 columns of data
    categorical target -- 11 animal outcomes (adopted, transferred, etc.)
    '''
    data = pd.read_csv("./data/Austin_Animal_Center/animal_outcome.csv").to_numpy()
    num_classes = len(np.unique(data[:,-1]))

    # onehot encode
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    data = encoder.fit_transform(data)

    # seperate into features and targets
    x = data[:,:-num_classes]
    y = data[:,-num_classes:]

    # convert into a torch tensor
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)

    # pad x to the next power of 2
    pad_size = int(2**np.ceil(np.log2(x.size(1)))) - x.size(1)
    x = F.pad(x, (0, pad_size))

    xtrain, ytrain, xtest, ytest = train_test_split(x, y)
    info = InfoData(8, 11, True)
    return xtrain, ytrain, xtest, ytest, info

def load_parkinsons():
    '''
    n = 5875
    regression features -- 16 columns of data TODO: doesn't match fastfood paper
        - the features represent biometical voice measures.
        - 42 subjects, each with an average of 140 samples
    regression target1 -- domain = [7.0, 54.992]
    regression target2 -- domain = [5.0377, 39.511]
    '''
    data = pd.read_csv("./data/Parkinson_Telemonitor/parkinsons_updrs.data")
    
    features = data[['Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
       'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
       'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']]
    target1 = data['total_UPDRS']
    target2 = data['motor_UPDRS']

    x = features.to_numpy().astype(float)
    x = torch.from_numpy(x).to(device).to(torch.float)

    y1 = target1.to_numpy().astype(float)
    y1 = torch.from_numpy(y1).to(device).to(torch.float)

    y2 = target2.to_numpy().astype(float)
    y2 = torch.from_numpy(y2).to(device).to(torch.float)
    
    xtrain, ytrain, xtest, ytest = train_test_split(x, y)
    info = InfoData(16, 1, False)
    return xtrain, ytrain, xtest, ytest, info


def load_red_wine_quality():
    '''
    n = 1599
    regression features -- 11 columns of data
    categorical target -- quality of wine on scale 0-10. Observed domain = [3, 4, 5, 6, 7, 8]
    '''
    data = pd.read_csv("./data/Wine_Quality/winequality-red.csv", delimiter=";").to_numpy().astype(float)
    x = torch.from_numpy(data[:,:-1]).to(device).to(torch.float32)
    # pad x to the next power of 2
    pad_size = int(2**np.ceil(np.log2(x.size(1)))) - x.size(1)
    x = F.pad(x, (0, pad_size))

    y = torch.from_numpy(data[:,-1]).to(device).to(torch.int64)
    # move targets onto range (0, num_classes)
    _, y = np.unique(data[:,-1], return_inverse=True)
    y = torch.from_numpy(y).to(device)

    xtrain, ytrain, xtest, ytest = train_test_split(x, y)
    info = InfoData(16, 6, True)
    return xtrain, ytrain, xtest, ytest, info


def load_white_wine_quality():
    '''
    n = 4898
    regression features -- 11 columns of data
    categorical target -- quality of wine on scale 0-10. Observed domain = [3, 4, 5, 6, 7, 8, 9]
    '''
    data = pd.read_csv("./data/Wine_Quality/winequality-white.csv", delimiter=";").to_numpy()
    x = torch.from_numpy(data[:,:-1]).to(device).to(torch.float32)
    # pad x to the next power of 2
    pad_size = int(2**np.ceil(np.log2(x.size(1)))) - x.size(1)
    x = F.pad(x, (0, pad_size))

    y = torch.from_numpy(data[:,-1]).to(device).to(torch.int64)
    # move targets onto range (0, num_classes)
    _, y = np.unique(data[:,-1], return_inverse=True)
    y = torch.from_numpy(y).to(device)
    
    xtrain, ytrain, xtest, ytest = train_test_split(x, y)
    info = InfoData(16, 7, True)
    return xtrain, ytrain, xtest, ytest, info


def load_insurance():
    '''
    n = 5821
    regression features -- 85 columns of data
    binary target 
    '''
    data = pd.read_csv("./data/Insurance_Company/tic_data.txt", delimiter="\t").to_numpy()
    x = torch.from_numpy(data[:,:-1]).to(device).to(torch.float32)
    # pad x to the next power of 2
    pad_size = int(2**np.ceil(np.log2(x.size(1)))) - x.size(1)
    x = F.pad(x, (0, pad_size))
    y = torch.from_numpy(data[:,-1]).to(device).to(torch.int64)

    xtrain, ytrain, xtest, ytest = train_test_split(x, y)
    info = InfoData(128, 2, True)
    return xtrain, ytrain, xtest, ytest, info


def load_CT_slices():
    '''
    n = 53500
    74 patients; (43 male, 31 female)
    regression features -- 384 columns of data
        - 2 - 241:         Histogram describing bone structures
        - 242 - 385:       Histogram describing air inclusions
    regression target
        - Values are in the range [0; 180] where 0 denotes
	      the top of the head and 180 the soles of the feet.
    '''
    data = pd.read_csv("./data/location_of_CT_slices/slice_localization_data.csv").to_numpy()
    x = torch.from_numpy(data[:,1:-1]).to(device)
    y = torch.from_numpy(data[:,-1]).to(device)
    # pad x to the next power of 2
    pad_size = int(2**np.ceil(np.log2(x.size(1)))) - x.size(1)
    x = F.pad(x, (0, pad_size))

    xtrain, ytrain, xtest, ytest = train_test_split(x, y)
    info = InfoData(512, 1, False)
    return xtrain, ytrain, xtest, ytest, info


def load_KEGG_network():
    '''
    n = 65553
    regression features -- 27 columns of data
    TODO: what is the target?
    '''
    data = pd.read_csv("./data/KEGG_Metabolic_Network/Reaction Network (Undirected).data", header=None)
    data = data.iloc[:,1:]
    data = data.replace("?", pd.NA).dropna()
    data = np.array(data, dtype=float)

    x = torch.from_numpy(data[:,:-1]).to(device)
    y = torch.from_numpy(data[:,-1]).to(device)
    # pad x to the next power of 2
    pad_size = int(2**np.ceil(np.log2(x.size(1)))) - x.size(1)
    x = F.pad(x, (0, pad_size))

    xtrain, ytrain, xtest, ytest = train_test_split(x, y)
    info = InfoData(32, None, None) # target variable is unknown
    return xtrain, ytrain, xtest, ytest, info


def load_year_prediction_MSD():
    '''
    n = 515,344
    regression features -- 90 columns of data
    categorical target -- year of release (91 unique values)
    '''
    data = pd.read_csv("./data/Year_Prediction_MSD/YearPredictionMSD.txt", header=None).to_numpy()
    x = torch.from_numpy(data[:,1:]).to(device)
    y = torch.from_numpy(data[:,0]).to(device)
    # pad x to the next power of 2
    pad_size = int(2**np.ceil(np.log2(x.size(1)))) - x.size(1)
    x = F.pad(x, (0, pad_size))

    xtrain, ytrain, xtest, ytest = train_test_split(x, y)
    info = InfoData(128, 91, True)
    return xtrain, ytrain, xtest, ytest, info

