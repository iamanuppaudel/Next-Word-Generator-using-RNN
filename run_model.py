import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
model = tf.keras.models.load_model("model")
tokenizer = Tokenizer()
seed_text= 'the more wounded cato is the better I wonder how foxface is making out oh shes fine i say peevishly im still angry she thought of hiding in'
#the Cornucopia and I didnt

def generate_text_seq(model, tokenizer, text_seq_length, seed_text, n_words):
    text=[]
    for _ in range(n_words):
        encoded=tokenizer.texts_to_sequences([seed_text])[0]
        encoded = pad_sequences([encoded], maxlen=5, truncating='pre')

        y_predict= model.predict_classes(encoded)
        predicted_word=''
        for word, index in tokenizer.word_index.items():
            if index== y_predict:
                predicted_word = word
                break
        seed_text= seed_text + ' '+ predicted_word
        text.append(predicted_word)
    return (' '.join(text))

generate_text_seq(model, tokenizer, 5, seed_text, 25)
