import pandas as pd
import numpy as np
import random
import csv
new_data=list()

lines = csv.reader(open("crop_final.csv", "r"))
data = list(lines)
#data=data[:17161]
for i in range(len(data)):
	if data[i][-1]=='12':
		new_data.append(data[i])
print(new_data)


"""
if data.Value == 0:
	new_data.append(data)
"""	

def splitDataset(dataset, splitRatio):
	#67% training size
	trainSize = int(len(dataset) * splitRatio);
	trainSet = []
	copy = list(dataset)
	print(len(copy))
	while len(trainSet) < trainSize:
		#generate indices for the dataset list randomly to pick ele for training data
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]


training,test=splitDataset(new_data,0.67)

with open('training.csv','a') as writefile:
	writer=csv.writer(writefile)
	writer.writerows(training)

writefile.close()

with open('test.csv','a') as writefile:
	writer=csv.writer(writefile)
	writer.writerows(test)

writefile.close()









