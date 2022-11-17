#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 17:01:41 2022

@author: diego
"""

class model:
    def __init__(self, name, tune_pars, best_estimator):
        self.name = name
        self.tune_pars = tune_pars
        self.best_estimator = best_estimator