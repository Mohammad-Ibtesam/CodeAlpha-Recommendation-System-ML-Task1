import ML_Functionality

while True:
    
    x=input(ML_Functionality.Prompt)
    match x:
        case '1':
            ML_Functionality.Recommend_Songs()
        case '2':
            ML_Functionality.Recommend_Songs_with_current_dataset()
        case _:
            exit(0)