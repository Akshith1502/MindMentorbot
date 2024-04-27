import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from django.http import JsonResponse
from django.shortcuts import render
import pandas as pd

# Load the pipeline model and data
pipeline = pickle.load(open('Dataset/model.pkl', 'rb'))
data = pd.read_csv('Dataset/data.csv')
lemmatizer = WordNetLemmatizer()

# Define functions for chatbot functionality

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def predict_answer(question, model):
    return model.predict([question])[0]

def get_chatbot_response(user_input):
    cleaned_input = clean_up_sentence(user_input)

    # Predict answer based on user input
    predicted_answer = predict_answer(user_input, pipeline)

    response = f"Predicted Answer: {predicted_answer}"

    return response

# Define Django views

def chatbot(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input', '')
        response = get_chatbot_response(user_input)
        print(response)
        return JsonResponse({'response': response})
    else:
        return JsonResponse({'error': 'Invalid request method'})

def index(request):
    # Render the index.html file
    return render(request, 'index.html')
