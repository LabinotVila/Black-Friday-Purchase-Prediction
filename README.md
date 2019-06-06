# Black Friday: Data Mining

This project uses two algorithms specified below to predict how much a new user will spend, and whether he is going to spend below or above average, the dataset can be found here: https://www.kaggle.com/mehdidag/black-friday

##### Random Forest Regression
To predict the value of the attribute 'Purchase', we have used Random Forest Regression, whose RMSE *[Root Mean Squared Error]* is around 3100, which is considered decent. 

##### Random Forest Classification
To predict this boolean value *[0 - below average, 1 - above avreage]* we have used Classification, included the Kernel Support Vectore Machine, whose accuracy is around 80% which is considered decent as well.

##### Graphical User Interface
A simple application made with TKinter which makes use of the two .py files described above and displays the results, the DataFrame preview. It is made from these components:
- Import Dataset: the button which opens a popup window where you can select the dataset you want to work with.
- Prepare Dataset: the button which does specific operations on the selected DataFrame.
- Test and Train: the text field to specify what percentage of the dataset we want to train and test.
- Perform Regression | Classification: the button which applies the algorithm and tells the accurracy or RMSE error.
- Export Results: the button to export results and compare them with the actual testing DataFrame.

![image](https://user-images.githubusercontent.com/33487958/59030563-9d133e00-8861-11e9-8d5e-30e91a8789f9.png)
