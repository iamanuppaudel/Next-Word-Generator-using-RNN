#%%
#import all the libraries
import tensorflow as tf 
import pandas as pd
import string


#%%

#read the file line by line
with open ("dataset.txt") as myfile:
    for line in myfile:
        data = myfile.readlines()

#removing the intro part (ie: before line7)
data= data[7:]

#remove all the \n from the lines
data=list(map(lambda x:x.strip(),data))

#remove all the empty element from list ("")
while("" in data):
    data.remove("")


#join all the lines of list and make continuous text
data = " ".join(data)

#print(data)

#cleaning the text(only alphanumeric allowed)
def data_cleaner(doc):
    tokens = doc.split()
    table = str.maketrans('','', string.punctuation)
    tokens = [word.translate(table) for word in tokens] #remove punctuations
    tokens = [word for word in tokens if word.isalpha()] #remove special characters
    tokens = [word.lower() for word in tokens] #make all words lowercase
    return tokens


tokens= data_cleaner(data)

#print(tokens[:50])

#%%

print(len(tokens))
print(len(set(tokens)))

#%%
length = 5+1
lines=[]

for i in range(length, len(tokens)):
    seq= tokens[i-length:i]
    line= ' '.join(seq)
    lines.append(line)

#print(len(lines))
#lines[0]
#lines[1]

#%%
import numpy as np 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

#%%
tokenizer = Tokenizer()

tokenizer.fit_on_texts(lines)
sequences= tokenizer.texts_to_sequences(lines)
sequences= np.array(sequences)
X,y = sequences[:,:-1] , sequences[:,-1]
#X[0]
#y[0]


# %%

vocabulary_size= len(tokenizer.word_index) +1 
y= to_categorical(y, num_classes=vocabulary_size)


# %%
model = Sequential()
#here both output_dim & input_length will be X.shape[1]
model.add(Embedding(vocabulary_size,5, input_length=5))

model.add(LSTM(100, return_sequences=True))
model.add(LSTM(120))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocabulary_size, activation='softmax'))

model.summary()
# %%
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X,y, batch_size=50, epochs=25)
model.save('model')

# %%
