from flask import Flask, render_template, request
import random
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the model and define helper functions
model = tf.keras.models.load_model('textgenerator.model')

# Define the text generation functions
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature):
    # Your text generation logic here
    return generated_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    length = int(request.form['length'])
    temperature = float(request.form['temperature'])
    generated_text = generate_text(length, temperature)
    return render_template('result.html', generated_text=generated_text)

if __name__ == '__main__':
    app.run(debug=True)
