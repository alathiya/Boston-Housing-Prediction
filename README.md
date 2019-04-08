# Project: Predicting Boston Housing Prices

In this project, I have used Boston housing dataset from UCI Machine Learning Repository to predict the house price in monetary value.
First the model is trained on housing data with feature inputs and target price using GridSearchCV Decision Tree Regressor
(Supervised Classification). After training model is tested and evaluated using performance metrics R2 (coefficient of determination). 
Some of the data exploration and feature observations steps are done before model training to gain insight into data stats and type.
Model performance analysis is done plotting learning curves for training and testing scores. This helps to understand underfitting 
and overfitting with respect to hyperparameter used.    




## Install



This project requires **Python** and the following Python libraries installed:



- [NumPy](http://www.numpy.org/)

- [Pandas](http://pandas.pydata.org/)

- [matplotlib](http://matplotlib.org/)

- [scikit-learn](http://scikit-learn.org/stable/)



You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)




## Data



The modified Boston housing dataset consists of 489 data points, with each datapoint having 3 features. 
This dataset is a modified version of the Boston Housing dataset found on the [UCI Machine Learning Repository]
(https://archive.ics.uci.edu/ml/datasets/Housing).



**Features**

1. `RM`: average number of rooms per dwelling

2. `LSTAT`: percentage of population considered lower status

3. `PTRATIO`: pupil-teacher ratio by town

**Target Variable**

4. `MEDV`: median value of owner-occupied homes


## Implementation 
Model is fitted using Decision Tree Algorithm. To get the optimized model, grid search technique is used to optimize max_depth
parameter for the decision tree. 
Implementation is using ShuffleSplit for an alternative form of cross validation. Optimal model is retrieve through best_estimate 
to predict housing price on test data. Predicted price is compared against data statistic to validate if predicted price justifies
the stats analysed earlier. At last applicability of model in real world setting is discussed.  