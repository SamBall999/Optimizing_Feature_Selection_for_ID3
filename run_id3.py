# main class

import pandas as pd
from id3 import id_3 # better way to do this?


# how best to store these features and target?
def read_text_file(filename):
    data_file = open(filename,"r")
    data = []
    for line in data_file.readlines():
        data.append(line)
    print(len(data)) # 5000 training examples
    print(len(data[0])) # 108 -> 100 features, letters in True, spaces and \n -> encode true and false and 1 and 0?
    print(data[0])
    data_file.close()
    return data


# alternative - use pandas??
def read_text_file_alt(filename):
    print("\nReading in training dataset...")
    data = pd.read_csv(filename, header = None) 
    # first separate features and targets
    features_and_target = data[0].str.split(pat=" ", expand=True)
    data["Features"] = features_and_target[0]
    data["Target"] = features_and_target[1]
    # split features into their own columns
    split_features = data["Features"].str.split(pat="", expand=True)
    for i in range(split_features.shape[1]-1):
        data[i] = split_features[i] # does this index the column??

    data.drop(columns =["Features"], inplace = True) # drop old features column
    print(data.head())
    return data




def main():
    print("Feature Optimization for ID3")
    data = read_text_file_alt("./Data/Training_Data.txt")
    #entropy(data)
    #info_gain_A = get_information_gain(data, 95) # must it be string or number
    #print(info_gain_A)
    id_3(data)

if __name__ == "__main__":
    main()