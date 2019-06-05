# PAMJA FILLESTARE
# importimi i librarive kryesore
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# lexo fajllin
data = pd.read_csv("black_friday.csv")

# shikimi i pare i shperndarjes se blerjeve: pershtypje e siperfaqes Gausiane
plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,7))
sns.distplot(data.Purchase, bins = 30, norm_hist=False, kde=False)
plt.xlabel('Sasia e shpenzuar ne Purchase')
plt.ylabel('Numri i bleresve')
plt.title('Shperndaraj e sasive te blerjeve')

# kontrollimi i formateve | per analize te metutjeshme
numeric_features = data.select_dtypes(include=[np.number])
numeric_features.dtypes

# pjesa qe perkthen vleren numerike ne atributin 'Purchase' ne numra binare
import statistics
median = statistics.median(data['Purchase'])

def translate(value):
    if (value <= median):
        return 0
    else:
        return 1
data['Purchase']=data['Purchase'].apply(translate)

# shfaq vlerat unike ne kolona
for col in data.columns:
    print('{} elemente unike: {}'.format(col,data[col].nunique()))
'''
    - Shihet qe user_id dhe product_id jane ~5400 dhe ~3100 respektivisht, 
    por ky informacion ndihmon vetem ne raste te caktuara, ne pergjithesi 
    shkakton mangesi te shumta ne algoritem, per kete arsye nuk merren ne 
    konsiderate. Shembull kur ndikon: nese dihet perdoruesi perkates perpara 
    blerjes!
'''

# importimi i librarise imputer - mbushja me zero e variablave te NaN
from sklearn.impute import SimpleImputer
# vlerat qe mungojne [kategoria 2 dhe 3 kane shume vlera null]
data.isnull().sum()
# mbush vlerat null me 0 - shablloni
imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value=0)
imputer = imputer.fit(data.iloc[:, 9:11])
data.iloc[:, 9:11] = imputer.transform(data.iloc[:, 9:11])

# heqja e dy kolonave te para te datasetit [user dhe product]
data.drop(data.columns[[0,1]], axis=1, inplace=True)

'''
    - Per arsye se na duhet te fshijme outliers prej kolonave te caktuara, 
    na duhet qe dataseti te jete me vlera numbers e jo NaN (per arsye se 
    tek funksioni remove_outliers me poshte, kerkohen numra e jo stringje).
'''

# OUTELIERS
from pandas.api.types import is_numeric_dtype
def remove_outlier(df):
    low = .05
    high = .95
    quant_df = df.quantile([low, high])
    for name in list(df.columns):
        if is_numeric_dtype(df[name]):
            df = df[(df[name] > quant_df.loc[low, name]) & (df[name] < quant_df.loc[high, name])]
    return df

remove_outlier(data)
'''
    - Metoda IQR per fshirje te outliers: caktohen dy numra, ne rastin tone
    high dhe low, te cilet nese kolona perkatese eshte numer dhe eshte 
    nen numrin minimal * 0.05 dhe mbi numrin maksimal * 0.95 atehere shquhet 
    si nje outlier dhe fshihet nga dataseti jone.
'''

# ANALIZA
# matrica e variablave te pavarura
X = data.iloc[:, :-1] 
# matrica e variablave te varura
y = data.iloc[:, -1]
# korelacionet 
corr = numeric_features.corr()
print (corr['Purchase'].sort_values(ascending=False)[:10])
# paraqitja grafike e korelacioneve 
sns.heatmap(corr, vmax=.8,annot_kws={'size': 10}, annot=True);
'''
    - Nga grafiku shihet qarte se nuk ekziston ndonje korelacion qe mund te 
    na ndihmoj ne predikimin e blerjeve, pervec ne kategorite perkatese te 
    produkteve [product_category_1, product_category_2, product_category_3], qe 
    dote thote se multipleksimi i kategorive te lartpermendura ka ndikim 
    ne kolonen Purchase.
'''
# vizualizimi i analizes se te dhenave
# blerjet - raporti mashkull / femer
_, ax1 = plt.subplots(figsize=(11,7)) # (12, 7) - madhesia maksimale e konzoles
gender_count = X['Gender'].value_counts()
ax1.pie(gender_count, explode=(0,0),labels=['Mashkull','Femer'], autopct='%1.0f%%')
ax1.axis('equal') # e merr parasysh variablen figsize=(12,7)
plt.legend()
plt.show()
# mosha e blerjeve
plt.subplots(figsize=(11,7))
sns.countplot(data['Age'], hue=X['Gender'])
# qytetet [A, B dhe C ne Category_Count]
city_category_count = X['City_Category'].value_counts()
city_category_labels = X['City_Category'].unique()
_, ax3 = plt.subplots(figsize=(11,7))
ax3.pie(city_category_count,explode=(0, 0, 0), labels=city_category_labels, autopct='%1.1f%%')
ax3.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()
# shprendarja e personave neper qytete
plt.subplots(figsize=(11,7))
sns.countplot(X['City_Category'],hue=data['Age'])
# shperndaraj e occupation
plt.subplots(figsize=(11,7))
X['Occupation'].value_counts().sort_values().plot('bar')

# funksion per lehtesim te paraqitjeve
def plotting_view(group_by, column, plot_kind):
    plt.figure(figsize=(11,6))
    data.groupby(group_by)[column].sum().sort_values().plot(plot_kind)
    
plotting_view('Product_Category_1','Purchase','barh')
plotting_view('Product_Category_2', 'Purchase', 'barh')
plotting_view('Product_Category_3', 'Purchase', 'barh')


# grafiket e ndelidhjeve
sns.countplot(X.Occupation)
# who spends more, males or females
sns.countplot(X.Gender)
# by age
sns.countplot(X.Age)
# by marital status?
sns.countplot(X.Marital_Status)
# distrubuation by city category
sns.countplot(X.City_Category)


# converting gender to 0 or 1 respectively
gender_dict = {'F': 0, 'M': 1}
# apply the dictionary to the column
X['Gender'] = X['Gender'].apply(lambda line: gender_dict[line])
X['Gender'].value_counts()

# converting age to numerical values
age_dict = {'0-17': 0, '18-25': 1, '26-35': 2, '36-45': 3, '46-50': 4, '51-55': 5, '55+': 6}
# Giving Age Numerical values
X['Age'] = X['Age'].apply(lambda line: age_dict[line])
X['Age'].value_counts()

# converting City_Category to binary
city_dict = {'A': 0, 'B': 1, 'C': 2}
X['City_Category'] = X['City_Category'].apply(lambda line: city_dict[line])
X['City_Category'].value_counts()

# Stay_in_current_city_years to binary - dummy variables
# libary importation
from sklearn.preprocessing import LabelEncoder
# krijimi i objektit
le = LabelEncoder()
# variabla te reja
X['Stay_In_Current_City_Years'] = le.fit_transform(X['Stay_In_Current_City_Years'])
X = pd.get_dummies(X, columns=['Stay_In_Current_City_Years'])
X.dtypes


# MODELI
# importimi i librarise per te ndare csv fajllin dhe ndarja
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 300, criterion='entropy', random_state = 0)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

# ME SVC
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

# calculate accuracy
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, predictions)
print ("Saktesia e algoritmit tone:", accuracy * 100, "%")

# confusion matrix
confusion_matrix = pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)