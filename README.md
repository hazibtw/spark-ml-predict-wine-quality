# spark-ml-predict-wine-quality
Predicting wine quality

1) Download data file from below URL
   https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
  
2) classify the wines into good, bad, and normal based on their quality
   1–4 will be poor quality, 5–6 will be average, 7–10 will be great
  
3) build our model, let’s separate our data into testing and training sets.
   This will place 70% of the observations in the original dataset into train and the remaining 30% of the observations into test.

4) We will use the randomForest classification algorithm to build the model.

5) We achieved ~70% accuracy with a very simple model. It could be further improved by feature selection, and possibly by trying different values of mtry.

6) We can find best model with number of trees Using CrossValidator 

