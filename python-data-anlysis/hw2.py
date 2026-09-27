import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder 
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
import os
import csv

#random arrays of different sizes for the normal distribution
arrayS = np.random.normal(loc = 5, scale = 9, size = 100)
arrayM = np.random.normal(loc = 5, scale = 9, size = 1000) 
arrayB = np.random.normal(loc = 5, scale = 9, size = 10000)
arrayVB = np.random.normal(loc = 5, scale = 9, size = 100000)

#numpy array of means plotted and shown
x = [np.mean(arrayS), np.mean(arrayM), np.mean(arrayB), np.mean(arrayVB)]
arrX = np.array(x)
y = [100, 1000, 10000, 100000]
plt.plot(y, np.abs(arrX))
print(arrX)
plt.show()

#random arrays of different sizes for the second distribution
arrayS = np.exp(-np.power(np.random.uniform(10, 20, 100), 2))
arrayM = np.exp(-np.power(np.random.uniform(10, 20, 1000), 2))
arrayB = np.exp(-np.power(np.random.uniform(10, 20, 10000), 2))
arrayVB = np.exp(-np.power(np.random.uniform(10, 20, 100000), 2))

#numpy array of means plotted and shown
x = [np.mean(arrayS), np.mean(arrayM), np.mean(arrayB), np.mean(arrayVB)]
arrX = np.array(x)
y = [100, 1000, 10000, 100000]
plt.plot(y, np.abs(arrX))
print(arrX)
plt.show()

fileName = "groceries.csv"

#code from CSCI307-F25-HW2 with edited minSupport / minConfidence values
with open(fileName) as f:
    txt = f.read()
    lines = txt.splitlines()#split here
    transactions = []
    for line in lines:
        if(line):
            words = line.split(',')
            transactions.append(words)

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
minSupport = 0.043 
itemsets = fpgrowth(df, min_support=minSupport, use_colnames=True)
minConfidence = 0.1
rules = association_rules(itemsets, metric="confidence", min_threshold=minConfidence)
print(rules)

fileName = "compsent-all-data.csv"

#lines using csv to create an array with values from the table in the file
with open(fileName, newline='') as f:
    reader = csv.reader(f)
    data = [row for row in reader if row]

#going through and making a list of all of the items that use Python as object_a or object_b
pythonData = []
for i in range (0, len(data)):
    if(data[i][2] == "Python" or data[i][3] == "Python"):
        pythonData.append(data[i])

#turning phrase into tokened transaction and taking out words that aren't useful
lanTransactions = []
newTrans = []
for i in range (0, len(pythonData)): #17
    phrase = pythonData[i][17]
    words = phrase.split()
    for word in words:
        tWord = word.lower()
        if (tWord != "the") and (tWord != "and") and (tWord != "is") and (tWord != "to") and (tWord != "a") and (tWord != "than") and (tWord != "be") and (tWord != "in") and (tWord != "of") and (tWord != "for") and (tWord != "that") and (tWord!="or"):
            newTrans.append(tWord)
    lanTransactions.append(newTrans)
    newTrans = []

#code from CSCI307-F25-HW2 with edited minSupport / minConfidence values
te = TransactionEncoder()
te_ary = te.fit(lanTransactions).transform(lanTransactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
minSupport = 0.1
itemsets = fpgrowth(df, min_support=minSupport, use_colnames=True)
minConfidence = 0.1
rules = association_rules(itemsets, metric="confidence", min_threshold=minConfidence)
print(rules)
