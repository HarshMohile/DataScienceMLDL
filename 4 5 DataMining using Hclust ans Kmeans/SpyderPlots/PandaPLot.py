# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:00:52 2020

@author: Harsh
"""

import pandas  as pd

cars = pd.read_csv("D:/360Assignments/Data Sets/Q7.csv")

cars.plot.bar()
cars['Points'].hist()
#cars['Points'].bar() 'Series' object has no attribute 'bar' WRONG
cars['Points'].plot.hist(bins=50)
cars['Points'].plot.bar()



cars.Points.mean()



    


