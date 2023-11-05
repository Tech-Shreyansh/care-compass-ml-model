from django.shortcuts import render
import pandas
from joblib import load
import numpy as np
from rest_framework import generics
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView

model = load('Savedmodels/ml-models.joblib')

class Predict(APIView):
    def post(self, request):
        response = request.data.get('response')
        numeric_response = [[41, 1, 0, 0, 4, 0, 0, 2, 1, 1]]
        # numeric_response = [[response['age'], response['gender'], response['stress_life_changes'], response['sleep_energy'], response['mood_emotions'], response['social_support'], response['isolation'], response['self_harm_thoughts'], response['eating_habits'], response['work_school_performance']]]
        model_fields = np.array(numeric_response,dtype=object).reshape((1,-1))
        return Response(round((model.predict(model_fields))[0]))