import numpy as np
from tkinter import *
import pandas as pd
from pandas import DataFrame
from tkinter.ttk import Combobox
from sklearn.impute import SimpleImputer
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import metrics
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import statistics
from sklearn.metrics import mean_squared_error
from tkinter import filedialog

frm_main = Tk()
frm_main.title("FIEK | Random Forest Prediction")

# NOTIFICATION
notify_message = StringVar()
notify_message.set("")
Label(frm_main, textvariable=notify_message).place(x=10, y=480)
ready = False

Label(frm_main, text="Preprocessing | do:", bg="yellow").place(x=25, y=85)

predicted_result = StringVar()
predicted_result.set("")
Label(frm_main, textvariable=predicted_result).place(x=275, y=190)

Label(frm_main, text="Test percentage size [ex. 20]").place(x=25, y=160)
txt_test = StringVar()
Entry(frm_main, textvariable=txt_test, width=25).place(x=25, y=190)

# DATASET ROWS AND COLUMNS LABEL
dataset_rows = StringVar()
dataset_columns = StringVar()
dataset_rows.set("")
dataset_columns.set("")
Label(frm_main, textvariable=dataset_rows).place(x=25, y=50)
Label(frm_main, textvariable=dataset_columns).place(x=130, y=50)

def show_demo(dataframe):
    global data, right_side
    
    try:
        right_side.destroy()
    except:
        right_side = Frame(frm_main).place(x=275,y=10, width=1525,height=490)
    
    partial = dataframe.head(10)
    
    i = 0
    for x in dataframe.columns:
        sick = Label(right_side, text=x[:15], bg="yellow").place(x=275 + (i * 125), y=10)
        i=i+1
    
    for k, v in partial.iterrows():
        for x in range (len(v)):
            sick = Label(right_side, text=v[x]).place(x=275 + (x * 125), y= 80 + (k * 40))
    

def import_dataset():    
    try:
        global data, dataset_rows, dataset_columns, median
        PATH = filedialog.askopenfilename()
        read = pd.read_csv (PATH) 
        data = DataFrame(read)
        dataset_rows.set("Rows: " + str(data.shape[0]))
        dataset_columns.set("Columns: " + str(data.shape[1]))
        median = statistics.median(data['Purchase'])
        
        notify_message.set("Dataset successfully imported") 
        
        show_demo(data)
    except:
        notify_message.set("An error occured!")

btn_import = Button(frm_main, text="Import Dataset", command=import_dataset, width=25).place(x=25, y=10)

# ------------------------ PREPROCESSING FUNCTIONS ---------------------------#

def prepare_dataset():
    try:
        global data, X, y
        data.isnull().sum()
        imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value=0)
        imputer = imputer.fit(data.iloc[:, 9:11])
        data.iloc[:, 9:11] = imputer.transform(data.iloc[:, 9:11])
        
        data.drop(data.columns[[0,1]], axis=1, inplace=True)
        
        low = .05
        high = .95
        quant_df = data.quantile([low, high])
        for name in list(data.columns):
            if is_numeric_dtype(data[name]):
                data[(data[name] > quant_df.loc[low, name]) & (data[name] < quant_df.loc[high, name])]  

        X = data.iloc[:, :-1] 
        y = data.iloc[:, -1]
        
        gender_dict = {'F': 0, 'M': 1}
        X['Gender'] = X['Gender'].apply(lambda line: gender_dict[line])
        X['Gender'].value_counts()
        
        age_dict = {'0-17': 0, '18-25': 1, '26-35': 2, '36-45': 3, '46-50': 4, '51-55': 5, '55+': 6}
        # Giving Age Numerical values
        X['Age'] = X['Age'].apply(lambda line: age_dict[line])
        X['Age'].value_counts()
        
        city_dict = {'A': 0, 'B': 1, 'C': 2}
        X['City_Category'] = X['City_Category'].apply(lambda line: city_dict[line])
        X['City_Category'].value_counts()
        
        le = LabelEncoder()
        
        X['Stay_In_Current_City_Years'] = le.fit_transform(X['Stay_In_Current_City_Years'])
        X = pd.get_dummies(X, columns=['Stay_In_Current_City_Years'])
        
        notify_message.set("Dataset successfully prepared.")
        
        show_demo(X)
    except:
        notify_message.set("Something went wrong!")
        
# -------------------------- ESTIMATE TREES ----------------------------------#
        
# assign max depth
Label(frm_main, text="Tree estimator:").place(x=25, y=290)
txt_estimators = StringVar()
Entry(frm_main, textvariable=txt_estimators, width=25).place(x=25, y=320)

test_col = StringVar()
train_col = StringVar()
test_col.set("")
train_col.set("")
Label(frm_main, textvariable=test_col).place(x=25, y=255)
Label(frm_main, textvariable=train_col).place(x=130, y=255)

def translate(value):
    if (value <= median):
        return 0
    else:
        return 1

def predict_boolean():
    try:
        global X_train, y_train, X_test, y_test, predictions
        
        y_train=y_train.apply(translate)
        y_test=y_test.apply(translate)
        
        est = int(txt_estimators.get())
        
        rf = RandomForestClassifier(n_estimators = est, random_state = 0)
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)
        
        accuracy = metrics.accuracy_score(y_test, predictions) * 100
        
        notify_message.set("Prediction successful, accuracy: " + str(round(accuracy)) + ".")
    except:
        notify_message.set("Prediction went wrong, terrible!")
        
    
def predict_value():
    try:
        global X_train, X_test, y_train, y_test, predictions
        
        est = int(txt_estimators.get())
        
        rf = RandomForestRegressor(n_estimators=est, random_state = 0)
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)
                
        rms = sqrt(mean_squared_error(y_test, predictions))
        
        notify_message.set("Prediction successful, mean error: " + str(round(rms)))
    except:
        notify_message.set("Predict did not go well, re-check information.")

def split_and_test():
    try:
        global X, y, X_train, X_test, y_train, y_test, test_col, train_col
        
        test = int(txt_test.get())/100
        test = float(test)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test, random_state=0, shuffle=False)
        
        test_col.set("Train: " + str(X_test.shape[0]))
        train_col.set("Test: " + str(X_train.shape[0]))
        
        notify_message.set("Split and train success!")
    except:
        notify_message.set("Split and train non success!")
        
def result_export():
    try:
        global predictions
        predictions.tofile('predictions.csv',sep='\n',format='%10.5f')
    except:
        print("A")        

Button(frm_main, text="Prepare Dataset", command=prepare_dataset, width=25).place(x=25, y = 120)
Button(frm_main, text="Test And Train", command=split_and_test, width=25).place(x=25,y=220)
Button(frm_main, text="Perform Regression", command=predict_value, width=25).place(x=25, y=360)
Button(frm_main, text="Perform Classification", command=predict_boolean, width=25).place(x=25, y=400)
Button(frm_main, text="Export Results", command=result_export, width=25).place(x=25, y=440)

# KEEP FRAME UP
frm_main.geometry("1900x510")
frm_main.resizable(True, False)
frm_main.mainloop()