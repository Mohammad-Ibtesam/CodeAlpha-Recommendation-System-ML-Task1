import numpy as np
import pickle
import csv
import random

Title = '\n\n\t\tSPOTIFY\'S SONG RECOMMENDATION SYSTEM\n\n\n'
Prompt = "\n\tPress:\n\t1. Recommend new songs.\n\t2. Recommend new songs with current dataset (if changed).\n\tAny key to quit.\n\n"
Number_of_genres = 3 # Pop, Hip-hop, and Rap
Desired_Accuracy = 75
Model = 'model.pkl'
Dataset = "Dataset.csv"
Songs_list = 'Songs list.csv'
Number_of_recommendations = 7
Liked_ratio = 0.8
Disliked_ratio = 0.3



def Load_Usernames():
    with open(Dataset, "r") as file:
        reader = csv.reader(file)
        first_line = next(reader)  
    data_array = list(first_line)
    return data_array[1::2]

def Load_Song_names():
    with open(Songs_list, "r") as file:
        reader = csv.reader(file)
        next(reader)
        next(reader)
        Data = [row[1:] for row in reader] 
        return np.transpose(Data)

def Load_Data():

    with open(Dataset, "r") as file:
        reader = csv.reader(file)
        next(reader)
        next(reader)
        Data = [list(map(int, row[1:])) for row in reader] 
    
    return Data


def Check(element):
    if 0 <= element[0] <= 100:
        if 0 <= element[1] <= 1:
            return
    print("The dataset provided is invalid. Kindly, provide the valid data.\n")
    exit()


def Transform_Data(Data):
    
    Number_of_users=int(len(Data[0])/2)
    Songs_per_genre=int(len(Data)/Number_of_genres)
    if Number_of_users<1 or Songs_per_genre<1:
        print("The dataset provided is insufficient. Kindly, reconsider the dataset before generating model.\n")
        exit()
    Compressed_Data=[[0 for _ in  range(Number_of_users)] for _ in  range(Number_of_genres)]
    for i in range(Number_of_genres):
        for j in range(Number_of_users):
            for k in range(Songs_per_genre):
                Check(Data[i*Songs_per_genre+k])
                if Data[i*Songs_per_genre+k][1+j*2]==1:
                    Compressed_Data[i][j]+=120
                else:
                    Compressed_Data[i][j]+=Data[i*Songs_per_genre+k][j*2]

    Transformed_Data=np.linalg.svd(Compressed_Data, full_matrices=False)
    return Transformed_Data


def Generate_Model():
    
    Usernames=Load_Usernames()
    Data=Load_Data()
    Songs=Load_Song_names()
    
    Data=Transform_Data(Data)
    
    with open(Model,'wb') as file:
        pickle.dump(Data,file)
        pickle.dump(Usernames,file)
        pickle.dump(Songs,file)


def Predict():
    
    with open(Model, 'rb') as file:
        Data = pickle.load(file)
        Usernames = pickle.load(file)
    U, S, Vt = Data
    Total_singular_value=0
    for i in S:
        Total_singular_value+=i

    Accuracy=0
    Top_K_Singular_Values=[]
    while Accuracy<Desired_Accuracy:
        idx=0
        for k,i in enumerate(S):
            if S[idx]<i:
                idx=k
        Top_K_Singular_Values.append([S[idx],idx])
        Accuracy+=((S[idx]*100)/Total_singular_value)
        S[idx]=0
    Number_of_users=len(Usernames)
    Resultant=[[0 for _ in  range(Number_of_users)] for _ in  range(Number_of_genres)]
    for i in range(Number_of_genres):
        for j in range(Number_of_users):
            for k in Top_K_Singular_Values:
                Resultant[i][j]+=((U[i][k[1]]*k[0])*Vt[k[1]][j])
    
    Resultant = np.transpose(Resultant)
    for i in range(Resultant.shape[0]):  #for i in range(len(Resultant)):  
        total = np.sum(Resultant[i]) 
        if total == 0:
            continue  
        max_idx = np.argmax(Resultant[i])
        if Resultant[i, max_idx] / total >= Liked_ratio:
            Resultant[i] = 0
            Resultant[i, max_idx] = Number_of_recommendations
            continue
        min_idx = np.argmin(Resultant[i])
        if Resultant[i,min_idx] / total <= Disliked_ratio:
            Resultant[i,min_idx] = 0
        total = np.sum(Resultant[i])
        if total > 0:
            Resultant[i]*=(Number_of_recommendations / total)
    return Resultant


def Recommend_Songs():

    with open(Model, 'rb') as file:
        Data = pickle.load(file) 
        Usernames = pickle.load(file)
        Songs = pickle.load(file)
    Recommended_quanties=Predict()

    Number_of_users=len(Recommended_quanties)
    Song_recommendation = [[] for _ in range(Number_of_users)]
    for i in range(Number_of_users):
        for j in range(Number_of_genres):
            Genre=Songs[j]
            Recommended_quantity = min(int(Recommended_quanties[i][j]), len(Genre))
            Genre=sorted(Genre)
            Recommended_Songs = random.sample(Genre, Recommended_quantity)
            Song_recommendation[i].extend(Recommended_Songs)
    
    print(Title)
    for i in range(len(Usernames)):
        print('\t\t=====  '+Usernames[i]+'  =====\n\t')
        print(" | ".join(Song_recommendation[i]))
        print('\n\n')


def Recommend_Songs_with_current_dataset():
    Generate_Model()
    Recommend_Songs()