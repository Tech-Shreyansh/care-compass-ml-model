from django.shortcuts import render
from joblib import load
# Create your views here.

model = load('./Savedmodels/ml-models.joblib')
