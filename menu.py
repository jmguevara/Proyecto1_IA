"""
Menu module
"""
import pandas as pd
import reader as rdr
import decision_tree as dt
import standardization as std
running = True
dataset = []

while(running):
    print('Select an option:')
    print('1- Read csv file')
    print('2- Standardize dataset')
    print('3- Random forest')
    print('4- Cross validation')
    print('5- End program')
    selection = input()
    if(selection == '1'):
        try:
            rdr.read()
            dataset = rdr.get_matrix()
            print('File loaded')
            print('Print dataset?')
            print('1-Yes / 2-No')
            print_selection = input()
            if(print_selection=='1'):
                df = pd.DataFrame(dataset)
                print(df) 
        except:
            print('File data.cvs not found!')

    elif(selection =='2'):
        try:
            std.normalization()
            std.to_csv()
        except:
            print('Error, please try loading the file again')
    elif(selection =='5'):
        running = False
print('Exit')

"""
    elif(selection == '3'):
    elif(selection == '4'):
"""
