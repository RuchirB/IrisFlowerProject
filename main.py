# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

print(type(url))
print(type(dataset))

# shape prints (150,5) for 150 rows, 5 columns
print(dataset.shape)

# head shows first 20 rows
print(dataset.head(20))

# descriptions like mean and median
print(dataset.describe())

# class distribution shows how much data per class
print(dataset.groupby('class').size())

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# histograms
dataset.hist()
pyplot.show()

# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

# Split-out validation dataset. 20% for validation, 80% to train
array = dataset.values
#X is an array of ARRAYS of all the first four values (2D array)
#Basically the part before the comma represents rows, and the part after represents columns
#since rows is : its like go through every row from array
#since columns is 0:4 its take all the columns from array until the third index from array
X = array[: , 0:4]
#y is an array of all the classes
#: signifies index 0 and 4 signifies stepping by 4
y = array[: , 4]

print("The following are X and Y")
print(X)
print(y)

#X_train and Y_train are for preparing the model, X_Validation and Y_Validation are for later once we have decided
#what model to actually use
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

#An array of tuples, corresponding to ["model name", model]
#we later go through each of these models to check which is the best for our data
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []


#name is index one, model is actual model, index two in the array of tuples, models

for name, model in models:
	#the k-fold cross validation is a procedure used to estimate the skill of the model on new data.
	#n_splits=10 means it will split the data into 10
	kfold = StratifiedKFold(n_splits=10, random_state=1)
	#compiling the results of how good the model was in cv_results
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	#putting each result for each model into "results" array
	results.append(cv_results)
	names.append(name)
	#printing out, formatted: name of model, how good it was, (standard deviation)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


#since the SVC model is the best, based on the previous code, we will use that and fit it to our training data
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print(predictions)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))