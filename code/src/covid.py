# import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# convert string to numeric function
def convertToNumeric(data):
    for i in data.columns:
        data[i] = data[i].replace({'Yes': 1, 'No': 0})
    return data


# main function
def main():
    # read covid 19 csv file
    print("Loading Covid-19 Data...")
    covidData = pd.read_csv("../data/Covid Dataset.csv")
    print("Loading Completed!!!\n")
    # check data
    print("==Covid Data==")
    print(covidData)
    # check data type
    print("\n==Covid Data Information==")
    print(covidData.info())
    # count the total columns and rows in covid dataset
    print("\n==Data Shape==")
    print(covidData.shape)
    # convert the data type
    print("\nConverting data type string to numeric...")
    convertToNumeric(covidData)
    print("Converting Completed!!!")
    # check data type after converting
    print("\n==Covid Data Information after converting==")
    print(covidData.info())
    # Change the variable name because the variable name is long and it may be truncated when you draw a graph later
    covidData.rename(columns={'Family working in Public Exposed Places': 'Family working in Public'}, inplace=True)
    covidData.rename(columns={'Visited Public Exposed Places': 'Visited Public Places'}, inplace=True)
    # check null value
    print("\n==Check NULL Values==")
    print(covidData.isnull().sum())
    # Check the correlation of each response variable for the Covid-19 variable
    # control plot size
    plt.figure(figsize=(16, 8.8))
    # correlation heatmap
    print("\nGenerating correlation plot...")
    corrPlot = sns.heatmap(covidData.corr()[['COVID-19']],
                           annot=True,  # show scale
                           linewidths=.5)  # line width
    # set title
    plt.title("Correlation of all variables for COVID-19", fontsize=20)
    # save the correlation plot to corrplot.png file
    fig = corrPlot.get_figure()
    fig.savefig('../picture/corrplot.png')
    print("Correlation plot saved!!!")
    # Eliminate values that are not correlated with COVID-19
    print("\nRemoving unnecessary variables...")
    covidData.drop(['Wearing Masks', 'Sanitization from Market'],
                   axis=1,
                   inplace=True)
    print("Removing Completed!!!")
    # check data columns and rows after preprocessing
    print('\n==Data shape after preprocessing==')
    print(covidData.shape)
    # set response and explanatory variables
    # explanatory variable
    y = covidData['COVID-19']
    # response variables
    x = covidData.iloc[:, range(0, 18)]
    print("\nDetaching training data and test data...")
    # split train and test data (train size = 70, test size = 30)
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=.3, random_state=1)
    print("Detaching Completed!!!")
    # fit logistic regression model
    print("\nFitting Logistic Model...")
    modelLog = LogisticRegression()
    modelLog.fit(xTrain, yTrain)
    print("Fit Completed!!!")
    # accuracy of logistic model
    predLog = modelLog.predict(xTest)
    print("\nLogistic Model Report:\n", classification_report(yTest, predLog))
    # fit SVM model
    print("\nFitting SVM Model..")
    modelSvm = SVC(kernel='linear')
    modelSvm.fit(xTrain, yTrain)
    print("Fit Completed!!!")
    # accuracy of SVM model
    predSvm = modelSvm.predict(xTest)
    print("\nSVM Model Report: \n", classification_report(yTest, predSvm))
    # fit Decision Tree model
    print("\nFitting Decision Tree Model...")
    modelDecision = DecisionTreeClassifier()
    modelDecision.fit(xTrain, yTrain)
    print("Fit Completed!!!")
    # accuracy of decision tree model
    predDecision = modelDecision.predict(xTest)
    print("\nDecision Tree Model Report:\n", classification_report(yTest, predDecision))
    # Compare each model's accuracy
    print("******Compare Model Accuracy******\n")
    print("Decision Tree Model Accuracy: ", accuracy_score(yTest, predDecision))
    print("SVM Model Accuracy: ", accuracy_score(yTest, predSvm))
    print("Logistic Model Accuracy: ", accuracy_score(yTest, predLog))
    print("\n**********************************\n")
    # Graph the influence of each response variable on COVID-19 in the optimal model
    print("Generating variable importance plot...")
    plt.figure(figsize=(18, 10))
    plt.barh(x.columns, modelDecision.feature_importances_)
    # set title
    plt.title("Feature Importance for COVID-19", fontsize=20)
    # save the plot to importVariableToCovid19.png file
    plt.savefig('../picture/importVariableToCovid19.png')
    print("Variable importance plot generation completed!!!")
    print("\nAll actions have been completed and you can view the plots in the 'picture' directory!!!")


if __name__ == '__main__':
    main()
