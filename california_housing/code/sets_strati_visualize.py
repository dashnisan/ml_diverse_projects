#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dashnisan
"""
from functions import load_housing_data

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rootpath="/home/diego/Documents/EDUCATION/LEARNING/IT/MACHINE_LEARNING/ML_Handson_2022/workspace/1/wsp1/"
datapath = os.path.join(rootpath, "datasets/housing/")
outpath = os.path.join(rootpath, "out/2_end2end/")
codepath = os.getcwd

housing = load_housing_data(datapath)
housing.info()

income_cats = pd.cut(housing["median_income"],
                     bins=[0.,1.5,3.0,4.5,6.,np.inf],
                     labels=[1,2,3,4,5])
income_cats.hist(rwidth=1.2, facecolor='g', alpha=0.6)
plt.title("Stratified Median Income")
plt.xlabel("Median Income [10^3 USD]")
plt.ylabel("Frequency")
plt.xticks(np.arange(0.75, 5, step=0.25), minor=True)
fformat = "pdf"
fname = outpath + "median_income_stratified." + fformat
plt.savefig(fname, format=fformat)
#income_cats.head()


