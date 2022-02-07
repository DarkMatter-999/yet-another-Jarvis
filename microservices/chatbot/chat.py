import random
import json
import numpy as np

import nltk
from nltk.stem.porter import PorterStemmer

import torch
from torch import nn

stemmer = PorterStemmer()

def bag_of_words(tokenized_sentence, words):
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, word in enumerate(words):
        if word in sentence_words:
            #set 1 if word is in sentence-words
            bag[idx] = 1.0

    return bag

intents = json.loads(open("./microservices/chatbot/intents.json", "r").read())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FILE = "./bin/chatmodel.pth"
data = torch.load(FILE)

# print(data)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    # nn.Dropout(0.5),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    # nn.Dropout(0.5),
    nn.Linear(hidden_size, output_size),
    # nn.Softmax(),
).to(device)

model.load_state_dict(model_state)
model.eval()

botname = "Jarvis"

def chat(sentence):
    
    sentence = nltk.word_tokenize(sentence)
    x = bag_of_words(sentence, all_words)

    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)

    output = model(x)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probabilities = torch.softmax(output, dim=1)
    probability = probabilities[0][predicted.item()]

    if probability.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response = (random.choice(intent['responses'])).replace("{botname}", botname)
                return response
    else:
        return "I did not understand"
