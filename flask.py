from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

app = Flask(__name__)

model = Sequential()
model.add(Dense(32, input_shape=(11,)))
# Load models
model1 = tf.keras.models.load_model("Model1.h5")
model2 = tf.keras.models.load_model("Model2.h5")

predicted_labels = None

@app.route('/predict', methods=['POST', 'GET'])
def predict_asd():
    global predicted_labels
    data = request.json
    A1 = data["input"]["A1"]
    A2 = data["input"]["A2"]
    A3 = data["input"]["A3"]
    A4 = data["input"]["A4"]
    A5 = data["input"]["A5"]
    A6 = data["input"]["A6"]
    A7 = data["input"]["A7"]
    A8 = data["input"]["A8"]
    A9 = data["input"]["A9"]
    A10 = data["input"]["A10"]

    speech = A1 + A5 + A3
    social = A1 + A2 + A3 + A4 + A5 + A6 + A7 + A9 + A10
    sensory = A1 + A6 + A8 + A10
    physical = A7 + A8 + A9
    total = A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 + A10

    user_input = pd.DataFrame({
        "A1": [A1],
        "A2": [A2],
        "A3": [A3],
        "A4": [A4],
        "A5": [A5],
        "A6": [A6],
        "A7": [A7],
        "A8": [A8],
        "A9": [A9],
        "A10": [A10],
        "Speech": [speech],
        "Sensory": [sensory],
        "Physical": [physical],
        "Social": [social],
        "Total": [total],
        "Age_Years": [data["input"]["Age"]],
        "Sex": [data["input"]["Sex"]],
        "Jaudience": [data["input"]["Jaudience"]],
        "Family_mem_with_ASD": [data["input"]["Family_mem_with_ASD"]],
        "Who_completed_the_test": [data["input"]["Who_completed_the_test"]]
    })

    predictions = model1.predict(user_input)
    predicted_labels = (predictions > 0.4).astype(int)

    # Predict therapy
    user_input["ASD_traits"] = predicted_labels
    user_input = user_input[
        ["Speech", "Sensory", "Physical", "Social", "Total", "ASD_traits"]
    ]
    predictions = model2.predict(user_input)

    # Mengambil 3 hasil prediksi tertinggi
    top_n = 3
    top_indices = np.argsort(predictions[0])[::-1][:top_n]
    top_probabilities = predictions[0][top_indices]

    label_asli = {
        1: "Speech 1",
        2: "Speech 2",
        3: "Speech 3",
        4: "Social 1",
        5: "Social 2",
        6: "Social 3",
        7: "Sensory 1",
        8: "Sensory 2",
        9: "Sensory 3",
        10: "Physical 1",
        11: "Physical 2",
        12: "Physical 3",
    }

    def top_predictions():
        return [{"Therapy": label_asli.get(label, f'Unknown {label}'), "Probability": f'{prob:.4f}'} for label, prob in zip(top_indices, top_probabilities)]

    return jsonify({"prediction_asd": predicted_labels.tolist(), "top_predictions": top_predictions()})

if __name__ == '__main__':
    app.run(debug=True)
