from tkinter import *
from tkinter.ttk import *

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlsxwriter


r=Tk()
r.geometry("720x480")
r.title('Crop Prediction and PET')


frame=Frame(r)
frame.pack()

frame1=Frame(r)
frame1.pack()

frame2=Frame(r)
frame2.pack()

frame3=Frame(r)
frame3.pack()

frame4=Frame()
frame4.pack()

def back():
	#frame1.destroy()
	for child in frame1.winfo_children():
        	child.destroy()
        for child in frame2.winfo_children():
        	child.destroy()
	for child in frame3.winfo_children():
        	child.destroy()
	for child in frame4.winfo_children():
        	child.destroy() 
	button1=Button(frame1,text="Crop Accuracy",width=25,command=clickb1)
	button1.pack(padx=20,pady=(150,10))
	button2=Button(frame1,text="Crop Prediction",width=25,command=clickb2)
	button2.pack(padx=20,pady=10)
	button3=Button(frame1,text="PET",width=25,command=clickb3)
	button3.pack(padx=20,pady=10)

def clickb1():
	frame.destroy()
	for child in frame1.winfo_children():
        	child.destroy()
	acc=[94.61,87.06,96.10,94.99]
	label=['KNN','Logistic Regression','SVM_RBF','SVM_LINEAR']
	message=Message(frame3,text="Accuracy",width=320)
	message.pack()
        message1 = Message(frame3, text = label[0]+"  "+str(acc[0]) ,width=320) 
        message1.pack()
	message2 = Message(frame3, text = label[1]+"   "+str(acc[1]) ,width=320) 
        message2.pack()
	message3 = Message(frame3, text = label[2]+"  "+str(acc[2]) ,width=320) 
        message3.pack()
	message4 = Message(frame3, text = label[3]+"  "+str(acc[3]) ,width=320) 
        message4.pack()
	button10=Button(frame3,text="Back",width=25,command=back) 
        button10.pack()
	index = np.arange(len(acc))
	plt.bar(index, acc)
	plt.xlabel('Algorithm', fontsize=10)
	plt.ylabel('Percentage', fontsize=10)
	plt.xticks(index, label, fontsize=5, rotation=30)
	plt.title('Accuracy')
	plt.show()

	
	

	
def submit(e2,e3,e4,e5,e6,combo):
	    # Importing the dataset
	    
	    workbook = xlsxwriter.Workbook('results.xlsx')
	    worksheet = workbook.add_worksheet()
	    row =0
	    col =0

	    dataset = pd.read_csv('crop_final.csv')
	    X = dataset.iloc[:, [1,2,3,4]].values
	    y = dataset.iloc[:, 5].values
	    
	    ph=e2.get()
	    temp=e3.get()
	    rain=e4.get()
	    rh=e5.get()
	    area=e6.get()
	    previous=combo.current()	

	    # Splitting the dataset into the Training set and Test set
	    from sklearn.model_selection import train_test_split
	    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0, random_state = 0)

	    # Fitting Logistic Regression to the Training set
	    from sklearn.neighbors import KNeighborsClassifier
	    classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
	    classifier.fit(X_train,y_train)
	    row = 0
            col = 0
            print ("pH = "+str(ph))
            print ("Rainfall = "+str(rain))
           
           
            print ("Tempertaure = "+str(temp))
            print ("Relative Humidity = "+str(rh))
            print ("Area of land in hectres = "+str(area))
            print ("Last Crop = "+str(combo.get()))
            worksheet.write(row,col,float(str(ph)))
            worksheet.write(row,col+1,int(str(rain)))
            worksheet.write(row,col+2,int(str(temp)))
            worksheet.write(row,col+3,int(str(rh)))
            worksheet.write(row,col+4,float(str(area)))
            worksheet.write(row,col+5,str(previous))
            
            for child in frame1.winfo_children():
        	child.destroy()
            
            
            p=ph
            t=temp
            r=rain
            rel=rh
            size=str(area)
            size1 = int(size)
            #print(type(size1))
            #print(type(size1)
            previous_crop = previous
            n_cost_kg=119
            p_cost_kg=69
            k_cost_kg=169
           
            print(previous_crop)
            if previous_crop== 0:
                val=0
                n_lost=63 
                p_lost=28.5
                k_lost=16.5
                n_required=120
                p_required=60
                k_required=40
                yield1=3000
                net_n=size1*n_required-size1*n_lost
                net_p=size1*p_required-size1*p_lost
                net_k=size1*k_required-size1*k_lost
                
                
                ourMessage ="Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " kg and its cost is Rs." + str(round(net_n*n_cost_kg ,2))
                

                ourMessage1 ="Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " kg and its cost is Rs." + str(round(net_p*p_cost_kg ,2))
                

		ourMessage2 ="Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " kg and its cost is Rs." + str(round(net_k*k_cost_kg ,2))
                
                print("Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " and its cost is " + str(round(net_n*n_cost_kg ,2)))
                print("Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " and its cost is " + str(round(net_p * p_cost_kg,2)))
                print("Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " and its cost is " + str(round(net_k *k_cost_kg,2)))
                worksheet.write(row,col+6,str(net_n))
                worksheet.write(row,col+7,str(net_p))
                worksheet.write(row,col+8,str(net_k))
                worksheet.write(row,col+9,str(round(net_n*n_cost_kg ,2)))
                worksheet.write(row,col+10,str(round(net_p*n_cost_kg ,2)))
                worksheet.write(row,col+11,str(round(net_k*n_cost_kg ,2)))
                worksheet.write(row,col+12,yield1)
            elif previous_crop== 1:
                val=1
                n_lost=39
                p_lost=20
                k_lost=10.8
                n_required=120
                p_required=60
                k_required=40
                yield1=3000
                net_n=size1*n_required-size1*n_lost
                net_p=size1*p_required-size1*p_lost
                net_k=size1*k_required-size1*k_lost
                
                ourMessage ="Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " kg and its cost is Rs." + str(round(net_n*n_cost_kg ,2))
                

                ourMessage1 ="Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " kg and its cost is Rs." + str(round(net_p*p_cost_kg ,2))
                

		ourMessage2 ="Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " kg and its cost is Rs." + str(round(net_k*k_cost_kg ,2))
                
                print("Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " and its cost is " + str(round(net_n*n_cost_kg ,2)))
                print("Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " and its cost is " + str(round(net_p * p_cost_kg,2)))
                print("Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " and its cost is " + str(round(net_k *k_cost_kg,2)))    
                worksheet.write(row,col+6,str(net_n))
                worksheet.write(row,col+7,str(net_p))
                worksheet.write(row,col+8,str(net_k))
                worksheet.write(row,col+9,str(round(net_n*n_cost_kg ,2)))
                worksheet.write(row,col+10,str(round(net_p*n_cost_kg ,2)))
                worksheet.write(row,col+11,str(round(net_k*n_cost_kg ,2)))
                worksheet.write(row,col+12,yield1)
            elif previous_crop== 2:
                val=2
                n_lost=30   
                p_lost=15.8
                k_lost=11.3
                n_required=80
                p_required=40
                k_required=20
                yield1=2500
                net_n=size1*n_required-size1*n_lost
                net_p=size1*p_required-size1*p_lost
                net_k=size1*k_required-size1*k_lost
                ourMessage ="Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " kg and its cost is Rs." + str(round(net_n*n_cost_kg ,2))
                

                ourMessage1 ="Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " kg and its cost is Rs." + str(round(net_p*p_cost_kg ,2))
                

		ourMessage2 ="Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " kg and its cost is Rs." + str(round(net_k*k_cost_kg ,2))                
		print("Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " and its cost is " + str(round(net_n*n_cost_kg ,2)))
                print("Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " and its cost is " + str(round(net_p * p_cost_kg,2)))
                print("Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " and its cost is " + str(round(net_k *k_cost_kg,2)))   
                
                worksheet.write(row,col+6,str(net_n))
                worksheet.write(row,col+7,str(net_p))
                worksheet.write(row,col+8,str(net_k))
                worksheet.write(row,col+9,str(round(net_n*n_cost_kg ,2)))
                worksheet.write(row,col+10,str(round(net_p*n_cost_kg ,2)))
                worksheet.write(row,col+11,str(round(net_k*n_cost_kg ,2)))
                worksheet.write(row,col+12,yield1)
            elif previous_crop== 3:
                val=3
                n_lost=63
                p_lost=28.5
                k_lost=16.5
                n_required=120
                p_required=60
                k_required=40
                yield1=1200
                net_n=size1*n_required-size1*n_lost
                net_p=size1*p_required-size1*p_lost
                net_k=size1*k_required-size1*k_lost

		ourMessage ="Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " kg and its cost is Rs." + str(round(net_n*n_cost_kg ,2))
                

                ourMessage1 ="Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " kg and its cost is Rs." + str(round(net_p*p_cost_kg ,2))
                

		ourMessage2 ="Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " kg and its cost is Rs." + str(round(net_k*k_cost_kg ,2))
                print("Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " and its cost is " + str(round(net_n*n_cost_kg ,2)))
                print("Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " and its cost is " + str(round(net_p * p_cost_kg,2)))
                print("Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " and its cost is " + str(round(net_k *k_cost_kg,2)))
                
                worksheet.write(row,col+6,str(net_n))
                worksheet.write(row,col+7,str(net_p))
                worksheet.write(row,col+8,str(net_k))
                worksheet.write(row,col+9,str(round(net_n*n_cost_kg ,2)))
                worksheet.write(row,col+10,str(round(net_p*n_cost_kg ,2)))
                worksheet.write(row,col+11,str(round(net_k*n_cost_kg ,2)))
                worksheet.write(row,col+12,yield1)
            elif previous_crop== 4:
                val=4
                n_lost=140
                p_lost=70
                k_lost=41.1
                n_required=220
                p_required=130
                k_required=69
                yield1=3500
                net_n=size1*n_required-size1*n_lost
                net_p=size1*p_required-size1*p_lost
                net_k=size1*k_required-size1*k_lost

		ourMessage ="Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " kg and its cost is Rs." + str(round(net_n*n_cost_kg ,2))
                

                ourMessage1 ="Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " kg and its cost is Rs." + str(round(net_p*p_cost_kg ,2))
                

		ourMessage2 ="Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " kg and its cost is Rs." + str(round(net_k*k_cost_kg ,2))
                print("Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " and its cost is " + str(round(net_n*n_cost_kg ,2)))
                print("Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " and its cost is " + str(round(net_p * p_cost_kg,2)))
                print("Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " and its cost is " + str(round(net_k *k_cost_kg,2)))
                worksheet.write(row,col+6,str(net_n))
                worksheet.write(row,col+7,str(net_p))
                worksheet.write(row,col+8,str(net_k))
                worksheet.write(row,col+9,str(round(net_n*n_cost_kg ,2)))
                worksheet.write(row,col+10,str(round(net_p*n_cost_kg ,2)))
                worksheet.write(row,col+11,str(round(net_k*n_cost_kg ,2)))
                worksheet.write(row,col+12,yield1)
            elif previous_crop== 5:
                val=5
                n_lost=63
                p_lost=28.5
                k_lost=16.5
                n_required=135
                p_required=69
                k_required=58
                yield1=3100
                net_n=size1*n_required-size1*n_lost
                net_p=size1*p_required-size1*p_lost
                net_k=size1*k_required-size1*k_lost

		ourMessage ="Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " kg and its cost is Rs." + str(round(net_n*n_cost_kg ,2))
                

                ourMessage1 ="Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " kg and its cost is Rs." + str(round(net_p*p_cost_kg ,2))
                

		ourMessage2 ="Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " kg and its cost is Rs." + str(round(net_k*k_cost_kg ,2))
                print("Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " and its cost is " + str(round(net_n*n_cost_kg ,2)))
                print("Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " and its cost is " + str(round(net_p * p_cost_kg,2)))
                print("Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " and its cost is " + str(round(net_k *k_cost_kg,2)))
                worksheet.write(row,col+6,str(net_n))
                worksheet.write(row,col+7,str(net_p))
                worksheet.write(row,col+8,str(net_k))
                worksheet.write(row,col+9,str(round(net_n*n_cost_kg ,2)))
                worksheet.write(row,col+10,str(round(net_p*n_cost_kg ,2)))
                worksheet.write(row,col+11,str(round(net_k*n_cost_kg ,2)))
                worksheet.write(row,col+12,yield1)
            elif previous_crop== 6:
                val=6
                n_lost=24
                p_lost=8.7
                k_lost=8.1
                n_required=35
                p_required=50
                k_required=35
                yield1=900
                net_n=size1*n_required-size1*n_lost
                net_p=size1*p_required-size1*p_lost
                net_k=size1*k_required-size1*k_lost

		ourMessage ="Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " kg and its cost is Rs." + str(round(net_n*n_cost_kg ,2))
                

                ourMessage1 ="Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " kg and its cost is Rs." + str(round(net_p*p_cost_kg ,2))
                

		ourMessage2 ="Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " kg and its cost is Rs." + str(round(net_k*k_cost_kg ,2))
                print("Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " and its cost is " + str(round(net_n*n_cost_kg ,2)))
                print("Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " and its cost is " + str(round(net_p * p_cost_kg,2)))
                print("Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " and its cost is " + str(round(net_k *k_cost_kg,2)))
                worksheet.write(row,col+6,str(net_n))
                worksheet.write(row,col+7,str(net_p))
                worksheet.write(row,col+8,str(net_k))
                worksheet.write(row,col+9,str(round(net_n*n_cost_kg ,2)))
                worksheet.write(row,col+10,str(round(net_p*n_cost_kg ,2)))
                worksheet.write(row,col+11,str(round(net_k*n_cost_kg ,2)))
                worksheet.write(row,col+12,yield1)
            elif previous_crop == 7:
                val=7
                n_lost=42
                p_lost=28.5
                k_lost=21
                n_required=81
                p_required=64
                k_required=42
                yield1=3900
                net_n=size1*n_required-size1*n_lost
                net_p=size1*p_required-size1*p_lost
                net_k=size1*k_required-size1*k_lost

		ourMessage ="Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " kg and its cost is Rs." + str(round(net_n*n_cost_kg ,2))
                

                ourMessage1 ="Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " kg and its cost is Rs." + str(round(net_p*p_cost_kg ,2))
                

		ourMessage2 ="Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " kg and its cost is Rs." + str(round(net_k*k_cost_kg ,2))
                print("Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " and its cost is " + str(round(net_n*n_cost_kg ,2)))
                print("Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " and its cost is " + str(round(net_p * p_cost_kg,2)))
                print("Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " and its cost is " + str(round(net_k *k_cost_kg,2)))
                worksheet.write(row,col+6,str(net_n))
                worksheet.write(row,col+7,str(net_p))
                worksheet.write(row,col+8,str(net_k))
                worksheet.write(row,col+9,str(round(net_n*n_cost_kg ,2)))
                worksheet.write(row,col+10,str(round(net_p*n_cost_kg ,2)))
                worksheet.write(row,col+11,str(round(net_k*n_cost_kg ,2)))
                worksheet.write(row,col+12,yield1)
            elif previous_crop== 8:
                val=8
                n_lost=41
                p_lost=12
                k_lost=12
                n_required=120
                p_required=60
                k_required=60
                yield1=1500
                net_n=size1*n_required-size1*n_lost
                net_p=size1*p_required-size1*p_lost
                net_k=size1*k_required-size1*k_lost

		ourMessage ="Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " kg and its cost is Rs." + str(round(net_n*n_cost_kg ,2))
                

                ourMessage1 ="Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " kg and its cost is Rs." + str(round(net_p*p_cost_kg ,2))
                

		ourMessage2 ="Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " kg and its cost is Rs." + str(round(net_k*k_cost_kg ,2))
                print("Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " and its cost is " + str(round(net_n*n_cost_kg ,2)))
                print("Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " and its cost is " + str(round(net_p * p_cost_kg,2)))
                print("Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " and its cost is " + str(round(net_k *k_cost_kg,2)))
                worksheet.write(row,col+6,str(net_n))
                worksheet.write(row,col+7,str(net_p))
                worksheet.write(row,col+8,str(net_k))
                worksheet.write(row,col+9,str(round(net_n*n_cost_kg ,2)))
                worksheet.write(row,col+10,str(round(net_p*n_cost_kg ,2)))
                worksheet.write(row,col+11,str(round(net_k*n_cost_kg ,2)))
                worksheet.write(row,col+12,yield1)
            elif previous_crop== 9:
                val=9
                n_lost=75
                p_lost=37
                k_lost=162.5
                n_required=180
                p_required=60
                k_required=90
                yield1=2500
                net_n=size1*n_required-size1*n_lost
                net_p=size1*p_required-size1*p_lost
                net_k=size1*k_required-size1*k_lost

		ourMessage ="Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " kg and its cost is Rs." + str(round(net_n*n_cost_kg ,2))
                

                ourMessage1 ="Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " kg and its cost is Rs." + str(round(net_p*p_cost_kg ,2))
                
 
		ourMessage2 ="Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " kg and its cost is Rs." + str(round(net_k*k_cost_kg ,2))
                print("Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " and its cost is " + str(round(net_n*n_cost_kg ,2)))
                print("Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " and its cost is " + str(round(net_p * p_cost_kg,2)))
                print("Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " and its cost is " + str(round(net_k *k_cost_kg,2)))
                worksheet.write(row,col+6,str(net_n))
                worksheet.write(row,col+7,str(net_p))
                worksheet.write(row,col+8,str(net_k))
                worksheet.write(row,col+9,str(round(net_n*n_cost_kg ,2)))
                worksheet.write(row,col+10,str(round(net_p*n_cost_kg ,2)))
                worksheet.write(row,col+11,str(round(net_k*n_cost_kg ,2)))
                worksheet.write(row,col+12,yield1)
            elif previous_crop== 10:
                val=10
                n_lost=70
                p_lost=49
                k_lost=126
                n_required=250
                p_required=75
                k_required=190
                yield1=7000
                net_n=size1*n_required-size1*n_lost
                net_p=size1*p_required-size1*p_lost
                net_k=size1*k_required-size1*k_lost

		ourMessage ="Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " kg and its cost is Rs." + str(round(net_n*n_cost_kg ,2))
                

                ourMessage1 ="Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " kg and its cost is Rs." + str(round(net_p*p_cost_kg ,2))
                

		ourMessage2 ="Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " kg and its cost is Rs." + str(round(net_k*k_cost_kg ,2))
                print("Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " and its cost is " + str(round(net_n*n_cost_kg ,2)))
                print("Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " and its cost is " + str(round(net_p * p_cost_kg,2)))
                print("Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " and its cost is " + str(round(net_k *k_cost_kg,2)))  
                worksheet.write(row,col+6,str(net_n))
                worksheet.write(row,col+7,str(net_p))
                worksheet.write(row,col+8,str(net_k))
                worksheet.write(row,col+9,str(round(net_n*n_cost_kg ,2)))
                worksheet.write(row,col+10,str(round(net_p*n_cost_kg ,2)))
                worksheet.write(row,col+11,str(round(net_k*n_cost_kg ,2)))
                worksheet.write(row,col+12,yield1)
            elif previous_crop == 11:
                val=11
                n_lost=32
                p_lost=14
                k_lost=19
                n_required=120
                p_required=60
                k_required=60
                yield1=3200
                net_n=size1*n_required-size1*n_lost
                net_p=size1*p_required-size1*p_lost
                net_k=size1*k_required-size1*k_lost

		ourMessage ="Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " kg and its cost is Rs." + str(round(net_n*n_cost_kg ,2))
                

                ourMessage1 ="Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " kg and its cost is Rs." + str(round(net_p*p_cost_kg ,2))
                

		ourMessage2 ="Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " kg and its cost is Rs." + str(round(net_k*k_cost_kg ,2))
                print("Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " and its cost is " + str(round(net_n*n_cost_kg ,2)))
                print("Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " and its cost is " + str(round(net_p * p_cost_kg,2)))
                print("Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " and its cost is " + str(round(net_k *k_cost_kg,2)))
                worksheet.write(row,col+6,str(net_n))
                worksheet.write(row,col+7,str(net_p))
                worksheet.write(row,col+8,str(net_k))
                worksheet.write(row,col+9,str(round(net_n*n_cost_kg ,2)))
                worksheet.write(row,col+10,str(round(net_p*n_cost_kg ,2)))
                worksheet.write(row,col+11,str(round(net_k*n_cost_kg ,2)))
                worksheet.write(row,col+12,yield1)
            elif previous_crop == 12:
                val=12    
                n_lost=14.4
                p_lost=3.2
                k_lost=13.6
                n_required=20
                p_required=60
                k_required=20
                yield1=788
                net_n=size1*n_required-size1*n_lost
                net_p=size1*p_required-size1*p_lost
                net_k=size1*k_required-size1*k_lost

		ourMessage ="Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " kg and its cost is Rs." + str(round(net_n*n_cost_kg ,2))
                

                ourMessage1 ="Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " kg and its cost is Rs." + str(round(net_p*p_cost_kg ,2))
                

		ourMessage2 ="Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " kg and its cost is Rs." + str(round(net_k*k_cost_kg ,2))
                print("Dear farmers, The amount of N to be added is " + str(round(net_n,2)) + " and its cost is " + str(round(net_n*n_cost_kg ,2)))
                print("Dear farmers, The amount of p to be added is " + str(round(net_p,2)) + " and its cost is " + str(round(net_p * p_cost_kg,2)))
                print("Dear farmers, The amount of k to be added is " + str(round(net_k,2)) + " and its cost is " + str(round(net_k *k_cost_kg,2))) 
                worksheet.write(row,col+6,str(net_n))
                worksheet.write(row,col+7,str(net_p))
                worksheet.write(row,col+8,str(net_k))
                worksheet.write(row,col+9,str(round(net_n*n_cost_kg ,2)))
                worksheet.write(row,col+10,str(round(net_p*n_cost_kg ,2)))
                worksheet.write(row,col+11,str(round(net_k*n_cost_kg ,2)))
                worksheet.write(row,col+12,yield1)
            #size2=size1.text()
            #previous=previous1.text()
            values_test=[[p,t,r,rel]]
            y_pred_test = classifier.predict(values_test)
            
            print(y_pred_test)
            if(y_pred_test==0):
		textmsg="Crop to be grown is Wheat"
                print("Crop is Wheat")  
                worksheet.write(row,col+13,'Wheat')
                worksheet.write(row,col+14,1625)
            elif(y_pred_test==1):
	       textmsg="Crop to be grown is Rice"
               print ("Crop is Rice \n ")
               worksheet.write(row,col+13,'Rice')
               worksheet.write(row,col+14,1470)
        
      
        
            elif(y_pred_test==2):
		textmsg="Crop to be grown is Maize"
                print ("Crop is Maize") 
                worksheet.write(row,col+13,'Maize')
                worksheet.write(row,col+14,1365)    
        
    
            elif(y_pred_test==3):
		textmsg="Crop to be grown is Green gram"
                print ("Crop is Green gram")
                worksheet.write(row,col+13,'Green Gram')
                worksheet.write(row,col+14,4000)
                

          
    
    
            elif(y_pred_test==4):
 		textmsg= "Crop to be grown is Pea"              
		print ("Crop is Pea")
                worksheet.write(row,col+13,'Pea')
                worksheet.write(row,col+14,1410)

            elif(y_pred_test==5):
		textmsg="Crop to be grown is pigeon pea"                
		print ("Crop is pigeon pea")     
                worksheet.write(row,col+13,'Pigeon Pea')
                worksheet.write(row,col+14,1410)
                
             
                
            elif(y_pred_test==6):
		textmsg="Crop to be grown is Sunflower"
                print ("Crop is Sunflower")
                worksheet.write(row,col+13,'SunFlower')      
                worksheet.write(row,col+14,3300)    
    
            elif(y_pred_test==7):
		textmsg="Crop to be grown is Onion"
                print ("Crop is Onion")
                worksheet.write(row,col+13,'Onion')
                worksheet.write(row,col+14,4000)
                
            
            elif(y_pred_test==8):
		textmsg="Crop to be grown is Millets"
                print ("Crop is Millets")
                worksheet.write(row,col+13,'Millets')
                worksheet.write(row,col+14,2000)
      
    
            elif(y_pred_test==9):
		textmsg="Crop to be grown is Potato"
                print ("Crop is Potato")
                worksheet.write(row,col+13,'Potato')
                worksheet.write(row,col+14,2200)
              
            
            elif(y_pred_test==10):
		textmsg="Crop to be grown is Sugarcane"
                print ("Crop is Sugarcane")
                worksheet.write(row,col+13,'Sugarcane')
                worksheet.write(row,col+14,400)
              
            
            elif(y_pred_test==11):
		textmsg="Crop to be grown  is Cotton"
                print ("Crop is Cotton")
                worksheet.write(row,col+13,'Cotton')
                worksheet.write(row,col+14,5000)
              
            elif(y_pred_test==12):
		textmsg="Crop to be grown is Soyabean"
                print ("Crop is Soyabean")
                worksheet.write(row,col+13,'Soyabean')
                worksheet.write(row,col+14,5000)
            workbook.close()
	    #ui
            messageVar = Message(frame2, text = ourMessage,width=320) 
            messageVar.config(bg='lightgreen') 
            messageVar.pack( )
            
            messageVar1 = Message(frame2, text = ourMessage1,width=320) 
            messageVar1.config(bg='lightgreen') 
            messageVar1.pack( )
		
	    messageVar2 = Message(frame2, text = ourMessage2,width=320) 
            messageVar2.config(bg='lightgreen') 
            messageVar2.pack( )  
            
	    msg=Message(frame2,text=textmsg,width=400)
	    msg.config(bg='lightgreen')		
	    msg.pack()	   
	    
            button9=Button(frame2,text="Back",width=25,command=back) 
            button9.pack()	           
	

def clickb2():
	frame.destroy()
	for child in frame1.winfo_children():
        	child.destroy()
	label1=Label(frame1,text="Crop Details")
	label1.grid(row=5,column=5)
	label2=Label(frame1,text="pH")
	label2.grid(row=15,column=2,padx=10,pady=10)
	e2=Entry(frame1)
	e2.grid(row=15,column=3)
	label3=Label(frame1,text="Temperature")
	label3.grid(row=15,column=5,padx=(5,0))
	e3=Entry(frame1)
	e3.grid(row=15,column=6)
	label4=Label(frame1,text="Rainfall")
	label4.grid(row=25,column=2,padx=5,pady=10)
	e4=Entry(frame1)
	e4.grid(row=25,column=3,padx=5,pady=10)
	label5=Label(frame1,text="Relative Humidity")
	label5.grid(row=25,column=5,padx=5,pady=10)
	e5=Entry(frame1)
	e5.grid(row=25,column=6,padx=5,pady=10)
	label6=Label(frame1,text="Area(Hectares)")
	label6.grid(row=35,column=2,padx=10,pady=10)
	e6=Entry(frame1)
	e6.grid(row=35,column=3,padx=5,pady=10)
	label7=Label(frame1,text="Last Crop")
	label7.grid(row=35,column=5,padx=5,pady=10)	
	combo = Combobox(frame1)
 	combo['values']=("Wheat","Rice","Maize","Green Gram","Pea","Pigeon Pea","Sunflower","Onion","Millet","Potato","Sugarcane","Cotton","Soybean")
 	combo.current(0)
 	combo.grid(row=35,column=6,padx=5,pady=10)
	
	
		
	button5=Button(frame1,text="Back",width=25,command=back)
	button5.grid(row=40,column=3,padx=10,pady=10)
	button6=Button(frame1,text="Submit",width=25,command=lambda:submit(e2,e3,e4,e5,e6,combo))
	button6.grid(row=40,column=6,padx=10,pady=10)

#pet	
from sklearn import linear_model
from sklearn.metrics import recall_score,mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt
from math import sqrt

caty=list()

from sklearn.svm import SVR

def findandprint(trainy,validy,testy,predic):
	print(" \t\t Validation \t\t Test")
	print("RMSE \t\t" + str(sqrt(mean_squared_error(trainy,validy)))+"\t\t"+ str(sqrt(mean_squared_error(testy,predic))))
	print("R Squared \t"+ str(r2_score(trainy,validy))+"\t"+ str(r2_score(testy,predic)))
	print("R \t\t"+ str(sqrt(r2_score(trainy,validy)))+"\t\t"+ str(sqrt(r2_score(testy,predic))))
	print("MAE \t\t" + str(mean_absolute_error(trainy,validy))+"\t"+ str(mean_absolute_error(testy,predic)))
	

def printgraph(Y_test,Y_pred):
	plt.scatter(Y_test,Y_pred)
	plt.xlabel("Actual Label")
	plt.ylabel("Predicted Label")
	plt.plot(np.unique(Y_test),np.poly1d(np.polyfit(Y_test,Y_pred,1))(np.unique(Y_test)))
	#plt.text(5.8,3.1,'R-Square=0.9')
	plt.show()
        


def clickb3():
	frame.destroy()
	#Train-Test Data
	C1Data=pd.read_excel(io='test.xlsx')

	train=pd.DataFrame(C1Data[:682])
	test=pd.DataFrame(C1Data[682:])

	#linear Regression
	reg=linear_model.LinearRegression()
	reg.fit(train.drop(['pet'],axis=1),train['pet'])
	jinx=list(test['pet'])
	storm=list(train[10:310]['pet'])
	#print('Coefficients: \n', reg.coef_)
	C1Output=reg.predict(test.drop(['pet'],axis=1))
	print("\n Using Linear Regression")
	print(C1Output)

	#print(reg.intercept_)
	#print(reg.coef_)

	for i in range(len(jinx)):
		caty.append(round(C1Output[i],1))
		


	valida=reg.predict(train[10:310].drop(['pet'],axis=1))
	findandprint(storm,valida,jinx,C1Output)
	


	#using svr-rbf
	clf=SVR(kernel='rbf',gamma='auto')
	clf.fit(train.drop(['pet'],axis=1),train['pet'])
	output=clf.predict(test.drop(['pet'],axis=1))
	print("\n Using SVR-rbf")
	valida=clf.predict(train[10:310].drop(['pet'],axis=1))
	print(output)
	findandprint(storm,valida,jinx,output)


	#using svr-linear
	clf=SVR(kernel='linear',gamma='auto')
	clf.fit(train.drop(['pet'],axis=1),train['pet'])
	output=clf.predict(test.drop(['pet'],axis=1))
	print("\n Using SVR-linear")
	valida=clf.predict(train[10:310].drop(['pet'],axis=1))
	print(output)
	findandprint(storm,valida,jinx,output)

	printgraph(jinx,caty)		


	
button1=Button(frame1,text="Crop Accuracy",width=25,command=clickb1)
button1.pack(padx=20,pady=(150,10))
button2=Button(frame1,text="Crop Prediction",width=25,command=clickb2)
button2.pack(padx=20,pady=10)
button3=Button(frame1,text="PET",width=25,command=clickb3)
button3.pack(padx=20,pady=10)

r.mainloop()
