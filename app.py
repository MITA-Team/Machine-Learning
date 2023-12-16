from flask import Flask, request, jsonify
from google.cloud import storage
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import uuid
import os

app = Flask(__name__)

model1 = tf.keras.models.load_model("Model1.h5")
model2 = tf.keras.models.load_model("Model2.h5")

predicted_labels = None
image_name = None
bucket_name = "mita-storage"

storage_client = storage.Client.from_service_account_json('./serviceAccountKey.json')

@app.route('/predict', methods=['POST', 'GET'])
def predict_asd():
    global predicted_labels
    global image_name
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
        "Social": [social],
        "Sensory": [sensory],
        "Physical": [physical],
        "Total": [total],
        "Age_Years": [data["input"]["Age"]],
        "Sex": [data["input"]["Sex"]],
        "Jaudience": [data["input"]["Jaudience"]],
        "Family_mem_with_ASD": [data["input"]["Family_mem_with_ASD"]],
        "Who_completed_the_test": [data["input"]["Who_completed_the_test"]]
    })

    predictions = model1.predict(user_input)
    predicted_labels = (predictions > 0.5).astype(int)
    def predicted_label():
        if predicted_labels[0] == 1:
            result_message = "Memiliki gejala ASD"
        else:
            result_message = "Tidak memiliki gejala ASD"
        
        return result_message
        
    # Predict therapy
    user_input["ASD_traits"] = predicted_labels
    user_input = user_input[
        ["Speech", "Social", "Sensory", "Physical", "Total", "ASD_traits"]
    ]

    def percentage_delay(A):
        categories = ['speech', 'social', 'sensory', 'physical']
        a_cat_list = [
            [1, 1, 1, 0],
            [0, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 0]
        ]

        total_cases = len(A)
        category_sums = {category: 0 for category in categories}

        for i in range(total_cases):
            current_a = A[i]

            if current_a == 1:
                current_a_cat = a_cat_list[i]
                for j in range(len(categories)):
                    category_sums[categories[j]] += current_a_cat[j]

        # Calculate the total sum across all categories
        total_sum = sum(category_sums.values())

        category_percentages = {category: (count / total_sum) * 100 for category, count in category_sums.items()}

        labels = list(category_percentages.keys())
        sizes = list(category_percentages.values())

        colors = ["#ff9999","#66b3ff","#99ff99","#ffcc99"]

        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        plt.axis('equal')  

        plt.title('Percentage Delay by Category')
        
        global image_name
        image_name = str(uuid.uuid4()) + '.png'
        
        # Save image
        plt.savefig(image_name)
        plt.close()

        blob = storage_client.bucket(bucket_name).blob(image_name)
        blob.upload_from_filename(image_name)

        os.remove(image_name)

        return image_name

    A = [A1, A2, A3, A4, A5, A6, A7, A8, A9, A10]
    percentage_delay(A)

    predictions = model2.predict(user_input)

    # Top 3 predictions
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
        top_results = []
        for label, prob in zip(top_indices, top_probabilities):
            original_label = label_asli.get(label, None)
            
            if original_label is not None:
                result = {"Therapy": original_label, "Probability": f'{prob:.4f}'}
                top_results.append(result)
        
        return top_results

    return jsonify({
        "prediction_asd": predicted_label(), 
        'image': f'https://storage.googleapis.com/{bucket_name}/{image_name}',
        "top_predictions": top_predictions()
    })

if __name__ == '__main__':
    app.run(debug=True)
