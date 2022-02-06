import random
import json
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Dataset

intents = json.loads(open("./microservices/chatbot/intents.json", "r").read())

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

all_words = []
tags = []
patternXtags = []

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)

    for patterns in intent["patterns"]:
        word = nltk.word_tokenize(patterns)
        all_words.extend(word)

        patternXtags.append((word, tag))

ignore_words = ["?", "!", ".", ",", "'"]
all_words = [stemmer.stem(word.lower()) for word in all_words if word not in ignore_words]

all_words = sorted(set(all_words))
tags = sorted(set(tags))

# print(all_words)
# print(tags)
# print(patternXtags)

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

train_x = []
train_y = []

for (pattern_sentence, tag) in patternXtags:
    bag = bag_of_words(pattern_sentence, all_words)

    train_x.append(bag)

    label = tags.index(tag)
    train_y.append(label)

train_x = np.array(train_x)
train_y = np.array(train_y)

# Dataset class
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(train_x)
        self.x_data = train_x
        self.y_data = train_y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Training parameters
batch_size = 8
input_size = len(train_x[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 2000
decay = 1e-6

print(input_size, "==", len(all_words))
print(output_size, "==", len(tags))

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

#loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=decay, momentum=0.9, nesterov=True)
optimizer = Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

print(f'final loss: {loss.item()}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "./bin/chatmodel.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')