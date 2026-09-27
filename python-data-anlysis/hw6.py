import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

#given code 

samples = 100

# x is the input feature
x = np.linspace(0, 10, samples)
# y is our noise-free target feature
z = x + np.sin(2*x)
# this is how the data would look like without any noise
#plt.plot(x, z)

y = z.copy()
# divide the indices into train and test indices
indices = np.arange(len(y))
indices_train, indices_test = train_test_split(indices, random_state=1, test_size = 0.5)

# only add random noise to the test set
y[indices_train] += np.random.random(size = len(indices_train))

# can you see how some data is not noisy while the test data is noisy?
#plt.plot(x, z)
#plt.scatter(x, y)



X = np.reshape(x, (len(x), 1))
X_train, X_test, y_train, y_test = X[indices_train], X[indices_test], y[indices_train], y[indices_test]

# TODO: This is a default MLPRegressor, use the hyperparameters to improve performance
regr = MLPRegressor(alpha = 0, learning_rate_init = .0001, max_iter = 10000000, activation = 'tanh', hidden_layer_sizes = (8, 8), tol = .0000000001)
regr.fit(X_train, y_train)

# TODO: add code to get training and testing error for comparison
y_pred = regr.predict(X_train)
y_predT = regr.predict(X_test)
train_mse = mean_squared_error(y_train, y_pred)
test_mse = mean_squared_error(y_test, y_predT)
plt.scatter(X_train, y_train)
plt.scatter(X_train, y_pred)
plt.show()

print(train_mse)
print(test_mse)


#test_mse = mean_squared_error(y_test, sgd_reg.predict(X_test))

# download the dataset
# check documentation here to learn more: https://huggingface.co/datasets/stanfordnlp/imdb

splits = {'train': 'plain_text/train-00000-of-00001.parquet', 
          'test': 'plain_text/test-00000-of-00001.parquet', 
          'unsupervised': 'plain_text/unsupervised-00000-of-00001.parquet'}
# df contains the training set as a pandas dataframe
df = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["train"])
# dftest contains the test set as a pandas dataframe
dftest = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["test"])

model = SentenceTransformer("all-MiniLM-L6-v2")

# Example of getting sentence embeddings
# Embedding is a fancy term for vector representations of text that we get from language models
# The sentences to encode

five = pd.DataFrame()
seven = pd.DataFrame()
nine = pd.DataFrame()
eleven = pd.DataFrame()
thirteen = pd.DataFrame()
fifteen = pd.DataFrame()

lengthVector = df['text'].str.len()
df['embedding'] = list(model.encode(df['text'].tolist()))


for i in range(0, len(df)):
    length = lengthVector[i]
    if(length <= 500):
        five = pd.concat([five, df.iloc[[i]]], ignore_index = True)
    if(length <= 700):
        seven = pd.concat([seven, df.iloc[[i]]], ignore_index = True)
    if(length <= 900):
        nine = pd.concat([nine, df.iloc[[i]]], ignore_index = True)
    if(length <= 1100):
        eleven = pd.concat([eleven, df.iloc[[i]]], ignore_index = True)
    if(length <= 1300):
        thirteen = pd.concat([thirteen, df.iloc[[i]]], ignore_index = True)
    if(length <= 1500):
        fifteen = pd.concat([fifteen, df.iloc[[i]]], ignore_index = True)

myList = [five, seven, nine, eleven, thirteen, fifteen, df]
accuracyList = [0, 0, 0, 0, 0, 0, 0]
curIndex = 0

for dfF in myList:
    texts = dfF['text'].tolist()
    response = dfF['label'].values
    embeddings = np.stack(dfF['embedding'].values)
    indices = np.arange(len(texts))
    indices_train, indices_test = train_test_split(indices, random_state=1, test_size = 0.5)
    X_train, X_test, y_train, y_test = embeddings[indices_train], embeddings[indices_test], response[indices_train], response[indices_test]
    print("I've made it to train")
    regr = SGDClassifier()
    regr.fit(X_train, y_train)
    y_predT = regr.predict(X_test)
    accuracy = accuracy_score(y_test, y_predT)
    print(accuracy)
    accuracyList[curIndex] = accuracy
    curIndex = curIndex + 1
    regrAgain = MLPClassifier(learning_rate_init = .001, max_iter = 500, activation = 'tanh', hidden_layer_sizes = (8,8), tol = .0000000001)
    regrAgain.fit(X_train, y_train)
    y_predAgain = regrAgain.predict(X_test)
    accuracyAgain = accuracy_score(y_test, y_predAgain)
    print(accuracyAgain)

plt.figure()
x = [500, 700, 900, 1100, 1300, 1500, len(df)]
plt.plot(x,accuracyList)
plt.show()
