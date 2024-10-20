import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import string


nltk.download('punkt')
nltk.download('wordnet')

ignoreLetters = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('chatbot_model.h5')

THRESHOLD = 0.95

def Main():
    print("Schreibe dem KI Bot")
    text = input()
    wordsList = nltk.word_tokenize(text)
    wordsList = [lemmatizer.lemmatize(word) for word in wordsList if word not in ignoreLetters]
    bag = [0] * len(words)
    for w in wordsList:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    res = model.predict(np.array([bag]))[0]
    resultsIndex = np.argmax(res)
    tag = classes[resultsIndex]

        # Check if model prediction is above threshold
    if res[resultsIndex] > THRESHOLD:
        for intent in intents['intents']:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                break
    print(f"Antwort: {response} | {res[resultsIndex]*100}%")
    Main()
    
    
    
Main()