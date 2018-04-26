from flask import Flask
from keras.models import model_from_json
import numpy as np
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"


@app.route("/predict")
def predict():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("weights.h5")

    # ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3',
    #   'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_I', 'Embarked_Q',
    #   'Embarked_S']

    X = np.array([
        30.,    # Age
        0.,     # SibSp
        0.,     # Parch
        7.6292, # Fare
        0.,     # Pclass_1
        0.,     # Pclass_2
        1.,     # Pclass_3
        1.,     # Sex_female
        0.,     # Sex_male
        0.,     # Embarked_C
        0.,     # Embarked_I
        1.,     # Embarked_Q
        0.      # Embarked_S
    ]).reshape(1, 13)

    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = loaded_model.predict(X)
    return "%.2f%%" % (score[0] * 100)