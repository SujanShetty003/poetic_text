# poetic_text

Shakespeare Text Generation with LSTM Neural Networks
This project demonstrates the use of Long Short-Term Memory (LSTM) neural networks to generate text similar to that of William Shakespeare. By training on Shakespeare's writings, the model learns to predict the next character in a sequence, allowing it to generate coherent text that mimics Shakespeare's style.

**Table of Contents**
Introduction
Installation
Loading Shakespeare Texts
Preparing Data
Building the Recurrent Neural Network
Helper Function
Generating Text
Results
Final Idea
Introduction
Recurrent neural networks (RNNs), particularly LSTMs, are highly effective for processing sequential data such as text. In this project, we will train an LSTM network to write text that resembles Shakespeare's work. By predicting the next character in a sequence based on the previous characters, our model can generate plausible Shakespearean text.

I**nstallation**
To run this project, you will need a Python environment with the following libraries installed:

TensorFlow
NumPy
Random

**You can install these dependencies using pip:**
pip install tensorflow numpy


**Loading Shakespeare Texts**
First, we need a substantial amount of Shakespeare's text to train our model. We will download and read the text directly into our script:
import tensorflow as tf

filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
text = text[300000:800000]  # Using a subset of the text for training

**Preparing Data**
We need to convert the text into numerical data for training the neural network:

characters = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {i, c for i, c in enumerate(characters)}

SEQ_LENGTH = 40
STEP_SIZE = 3

sentences = []
next_char = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i + SEQ_LENGTH])
    next_char.append(text[i + SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_char[i]]] = 1

**Building the Recurrent Neural Network
Next, we build and compile our LSTM model:**
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation, Dense, LSTM

model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
model.fit(x, y, batch_size=256, epochs=4)

**Helper Function
We use a helper function to sample the next character based on the model's predictions:**

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

**Generating Text
The following function generates text using the trained model:**

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    
    for i in range(length):
        x_predictions = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1

        predictions = model.predict(x_predictions, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_char = index_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

    return generated

**Results
Here are some examples of generated text with different temperatures:**
print(generate_text(300, 0.2))
print(generate_text(300, 0.4))
print(generate_text(300, 0.5))
print(generate_text(300, 0.6))
print(generate_text(300, 0.7))
print(generate_text(300, 0.8))

**Temperature 0.2:**
ay, marry, thou dost the more thou dost the mornish,
and if the heart of she gentleman, or will,
the heart with the serving a to stay thee,
i will be my seek of the sould stay stay
the fair thou meanter of the crown of manguar;
the send their souls to the will to the breath:
the wry the sencing with the sen

**Temperature 0.6:**
warwick:
and, thou nast the hearth by the createred
to the daughter, that he word the great enrome;
that what then; if thou discheak, sir.

clown:
sir i have hath prance it beheart like!


**Temperature 0.8:**
i hear him speak.
what! can so young a thordo, word appeal thee,
but they go prife with recones, i thou dischidward!
has thy noman, lookly comparmal to had ester,
and, my seatiby bedath, romeo, thou lauke be;
how will i have so the gioly beget discal bone.

clown:
i have seemitious in the step--by this low,

