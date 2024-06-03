# import numpy as np
# arr1=np.array([1,3,2,4,0,0])
# arr1[4:6]=5,6
# print(arr1)
# arr2 = np.array([[1, 2, 3, 4, 5],(2,3,4,4,6)])
# arr3 = np.array([[1, 2, 3, 4, 5],[2,3,4,4,6]])
# arr4=np.array([[1,2,34,4,4],[4,4,44,456,7]])
# arr5=np.array([arr3,arr4])
# print(arr2)
# print(arr5)
# print(type(arr2))
# arr8 = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

# print(arr8)

# variable.ndim is used to check no of dimensions
# 1d only rows 2d only columns and rows 3d array add depths 
# to the concepts of 2d array type(a),a.dtype

# cube=np.array([2,4,6],ndmin=3)
# print(cube)
# #created an array with 3 dimensions
# arr = np.array([[1,2,3,4,5], [6,7,3,9,10]])
# print(arr)
# print('2th element on 2nd row: ', arr[1,-4])
# arr7= np.array([[[1,2,3,4,5], [6,7,3,9,10]],[[11,12,13,14,15],[16,17,18,19,20]]])
# print(arr7[0,1,3]) #9
# import numpy as np
# arr = np.array([1, 2, 3, 4, 5, 6, 7])
# arr2 = np.array([[1, 2, 3, 4, 5, 6, 7],[8,9,10,11,12,13,14]])
# print(arr[0:5:2])# last item can be hidden
# print(arr[:4]) #upto 4 value
# print(arr[-3:-1])# it will print 5 6 only for the last index place the no 
# print(arr[::2])
# print(arr2[1, 1:4])
# print(arr2[0:2, 2])
# print(arr2[0:2,1:5])
# arrayname.dtype to know the type of array
# dtype='S' arr = np.array([1, 2, 3, 4], dtype='S')
# a=arrayname.astype('i')
import numpy as np
arr = np.array([-1,1])
arr2 = np.array([2,1])
arrn=np.array([[10,3,6],[23,1,10]])
arrb=np.array([[12,3],[2,3],[5,4]])
# z=arr2-arr
# print(z)
# z=z*2#this is scaler multiplication
# print(z)
# z+=1
# print(z)
# print(z.mean())
# std=z.std()
# print("this is the standard deviation of the function",std)
# z.max()
dot2=np.dot(arrn,arrb)
print(dot2)
dot=np.dot(arr,arr2)
print(dot)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
intercept=10
slope=1.5
x_values = np.linspace(0, 10, 11)
y_values = x_values*slope + intercept #slope = y=mx+b
plt.xlabel("data x")
plt.ylabel("ydata")
plt.title("slope intercept")
plt.plot(x_values, y_values)
plt.xticks(x_values)
plt.yticks(y_values)
plt.show()

pie=np.array([0,np.pi/2,np.pi])
y=np.sin(pie)
print(pie)
print(y)
# Generate an array of 10 evenly spaced values from 0 to 1 (inclusive)
values = np.linspace(0,2*np.pi, num=10)
y=np.sin(values)
p=np.array([values,y])
t=pd.DataFrame(p)
print(t)#it will show data horizentally

pandadata={'Values':values,'SIN_VALUES':y}
ndata=pd.DataFrame(pandadata)
ndata.to_csv('datacsv3.csv',index=False)
print(ndata)
plt.plot(values,y)
plt.show()
#vector subtraction means subtracting
# a array from a array

# x = arr.view()
# #view should be effected after changings made to
# #original array
# y=arr.copy()
# arr[0] = 42
# print(arr)
# print(y)
# print(x)
# x[0]=45
# print(x)
# print(arr)
# The original array SHOULD be affected by the changes made to the view.
# . base returns the none if attribute owns data ie copy owns data
import numpy as np
abb=np.array([1, 2, 3, 4], ndmin=5)
print(abb)
print('shape of array :', abb.shape)

axx = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13,14,15,16,17,18])
newarr = axx.reshape(6, 3)
newabb=axx.reshape(2,3,3)
print(newabb)
print(newarr)
# converting a multidimensional array into 1d using -1
# bbr = np.array([[1, 2, 3], [4, 5, 6]])

# for x in bbr:#here x is an array row []
#   print(x)
#   for y in x:
#     print(y)
# import numpy as np
# arr = np.array([[[1, 2, 3,4], [ 5, 6,7,8]], [[9,10,11,12], [13,14,15,16]]])
# axx = np.array([[1, 2, 3, 4, 5],[6,7,8,9,10]])
# for x in arr:
#    print(x)
#    for y in x:
#      print(y)
#      for z in y:
#        print(z)
# for x in np.nditer(arr): #Iterating on Each Scalar Element
#   print(x)
# for y in np.nditer(arr, flags=['buffered'], op_dtypes=['S']):
# #flags is set to buffered which allows more efficient access to
# # arrays data . op_dtypes datatype of elements you are iterating
# #over should be treated as string S
#   print(y)#The b before each number indicates that these are bytes objects representing strings in Python.
# for x in np.nditer(axx[:, ::2]):
#  print(x)
# for x in np.nditer(arr[:,:,::2]):
#   print(x)


# arr = np.array([[[1, 2, 3], [4, 5, 6]], [[10, 11, 12], [13, 14, 15]]])
# axx = np.array([[1, 2, 3], [4, 5, 6]])
# for val, vam in np.ndenumerate(arr):
#     print(f"Element at index {val} is {vam}")
# for val, vam in np.ndenumerate(axx):
#     print(f"Element at index {val} is {vam}")
# .concatenate for joining arrays
# arr1 = np.array([1, 2, 3])

# arr2 = np.array([4, 5, 6])

# arr9 = np.dstack((arr1, arr2))
# arr = np.vstack((arr1, arr2))
# arr4=np.hstack((arr1,arr2))
# print(arr4)
# print(arr)
# print("this is the depth array ",arr9)
import numpy as np
axx = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13,14,15,16,17,18])
arr5=np.array([[1,2,34],[12,13,1]])
arr6=np.array([[2,3,44],[5,56,9]])
# arr7=np.stack((arr5,arr6),axis=1)
# print(arr7)
# arrat=np.dstack((arr5,arr6))
# arrat2=np.vstack((arr5,arr6))
# arrat3=np.hstack((arr5,arr6))
# print("THIS IS THE DEPTH    \n",arrat)
# print("THIS IS THE VERTICAL \n",arrat2)
# print("THIS IS THE HORIZNTAL\n",arrat3)
newarr= np.array_split(axx,4)
print(newarr[0])
print(newarr[1])
print(newarr[2])
print(newarr[3])
import numpy as np
arr = np.array([[1, 2], [3, 4],[5,5],[56,6],[5, 6], [7, 8], [9, 10], [11, 12]])
newarr = np.array_split(arr, 3)
print(newarr)
#   H.SPLIT(), V.SPLIT(), D.SPLIT()
#  NP.WHERE(ARR) IS USED TO GET INDEX OF THAT DESIRED NO 
#write a program to find even no from 3d array
import numpy as np
arrat=np.array([[[22,24],[34,567]],[[5654,67],[456,4342]]])
even=[]
for x in range(arrat.shape[2]):
    for y in range(arrat.shape[1]):
        for z in range(arrat.shape[0]):
            if arrat[x,y,z]%2 == 0:
                even.append(arrat[x,y,z])
print("THIS IS THE DESIRED RESULT",even)
#three dimensions are excessed using arr[x,y,z]
#in range(array.shape[dim]) to atterate through each row
arr = np.array([6, 7, 8, 9])
x = np.searchsorted(arr, 7)
print(x)
#searchsorted method works on sorted on array and it returns index
#searchsorted method can help with sorting and search the place new items can be inserted
x = np.searchsorted(arr, 7,side='right')
print(x)
arrsearch = np.array([1, 3, 5, 7]);x = np.searchsorted(arrsearch, [2, 4, 6]);
print(x)

# import numpy as  np 
# arr=np.array([[34,545,31,56],[12,674,57,865]])
# print(np.sort(arr))
# arr2=np.array(['banana','apple','mango ','palm','dates'])
# arr2=np.sort(arr2)
# print(arr2)
# x=[[True,False,True,True],[False,True,False,False]]
# newarr=arr[x]
# print(newarr)

# #filter_arr = arr > 42 it will directly create a filter
# #filter_arr = arr % 2 == 0
# arr = np.array([41, 42, 43, 44])
# filter_arr = []

# for element in arr:
#   if element %2==0:
#     filter_arr.append(True)
#   else:
#     filter_arr.append(False)

# newarr = arr[filter_arr]

# print(filter_arr)
# print(newarr)

# import numpy as np
# arrat=np.array([[[22,24],[34,567]],[[5654,67],[456,4342]]])
# even=[]
# for x in arrat:
#     for y in x:
#         for z in y:
#             if z%2 == 0:
#                 even.append(z)
# even=np.sort(even)
# print("THIS IS THE DESIRED RESULT",even)

#==============================================================
#==============================================================

# import numpy as np
# random_integers = np.random.randint(1, 100, size=(5, 5))
# print(random_integers)
from numpy import random
arr=random.randint(1,344,(3,2))
print(arr)
arrfloat=random.rand() 
# np.random.rand() produces random numbers that are greater than or equal to 0 and less than 1.
print(arrfloat)
amm=random.rand(3,5)
print(amm)
#random.choice return one value from group of 
#random values 
x = random.choice([3, 5, 7, 9], size=(3, 5))
print(x)
#DATA DISTRIBUTION , RANDOM PERMUTATION .

from numpy import random
import numpy as np
y=random.choice([3,5,7,9,11],p=[0.2,0.3,0.3,0.2,0.0],size=(3,5))
print("ORIGINAL ARRAY : \n",y,"\n")
q=np.random.permutation(y)
print("PERMUTATION OF Y: \n",q,"\n")

np.random.shuffle(y)
print("SHUFFLED ARRAY \n",y,"\n")
#in shuffle function it changes the main array.
#It's often used when you want to keep the original
# data intact and create a new randomized version.


