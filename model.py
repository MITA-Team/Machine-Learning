import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

url = "https://github.com/MITA-Team/Machine-Learning/raw/main/Datasets.csv"
data = pd.read_csv(url)

# Klasifikasi ASD
# Menginisialisasi data label dan input
X = data.drop(["Therapy", "ASD_traits", "Percentage", "Case_No"], axis=1)
Y = data["ASD_traits"]

# Split data dengan Training data 90% dan validation data 10%
training_size = int(len(data) * 0.9)

X_train = X[:training_size]
X_test = X[training_size:]
Y_train = Y[:training_size]
Y_test = Y[training_size:]


# Model Klasifikasi ASD
model1 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='linear')
])
model1.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
history_classification=model1.fit(X_train, Y_train, epochs=10, batch_size=50, validation_data=(X_test, Y_test))


# Model Rekomendasi
# Menginisialisasi fitur dan label untuk sistem rekomendasi
label_encoder = LabelEncoder()
therapy = label_encoder.fit_transform(data["Therapy"])
features = data[["Speech", "Social", "Sensory", "Physical", "Total", "ASD_traits"]]

# Split data training 80% dan testing 20%
x_train = features[:training_size]
x_test = features[training_size:]
y_train = therapy[:training_size]
y_test = therapy[training_size:]

# Normalize data
x_train_norm = ((x_train * x_train.mean()) / x_train.std()).to_numpy()
x_test_norm = ((x_test * x_test.mean()) / x_test.std()).to_numpy()

# One hot encoding labels
y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=12)
y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes=12)

# Model Rekomendasi
model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(6,), activation="relu"),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(12, activation="softmax")])
model2.compile(optimizer='Adam', loss="categorical_crossentropy", metrics=["accuracy"])
history_reccomendations = model2.fit(
    x_train,
    y_train_encoded,
    epochs=10,
    batch_size=32,
    verbose=1,
    validation_data=(x_test, y_test_encoded),
)

input_A1, input_A2, input_A3, input_A4, input_A5 = 1, 1, 0, 1, 0
input_A6, input_A7, input_A8, input_A9, input_A10 = 0, 0, 1, 0, 1
input_speech = input_A1 + input_A5 + input_A3
input_social = (
    input_A1
    + input_A2
    + input_A3
    + input_A4
    + input_A6
    + input_A7
    + input_A9
    + input_A10
)
input_sensory = input_A1 + input_A6 + input_A8 + input_A10
input_physical = input_A7 + input_A8 + input_A9
input_total = (
    input_A1
    + input_A2
    + input_A3
    + input_A4
    + input_A5
    + input_A6
    + input_A7
    + input_A8
    + input_A9
    + input_A10
)
(
    input_Ages_Years,
    input_Sex,
    input_Jaudience,
    input_Family_mem_with_ASD,
    input_Who_completed_the_test,
) = (2, 1, 0, 0, 0)
user_input = pd.DataFrame(
    [
        [
            input_A1,
            input_A2,
            input_A3,
            input_A4,
            input_A5,
            input_A6,
            input_A7,
            input_A8,
            input_A9,
            input_A10,
            input_speech,
            input_social,
            input_sensory,
            input_physical,
            input_total,
            input_Ages_Years,
            input_Sex,
            input_Jaudience,
            input_Family_mem_with_ASD,
            input_Who_completed_the_test,
        ]
    ],
    columns=[
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
        "A8",
        "A9",
        "A10",
        "Speech",
        "Social",
        "Sensory",
        "Physical",
        "Total",
        "Age_Years",
        "Sex",
        "Jaudience",
        "Family_mem_with_ASD",
        "Who_completed_the_test",
    ],
)
print(user_input)
# Membuat prediksi pada data uji
predictions = model1.predict(user_input)
# Mengevaluasi hasil prediksi (berdasarkan threshold 0.5)
predicted_labels = (predictions > 0.5).astype(int)
print()
if predicted_labels==1:
    print(f"{predictions} menunjukkan gejala ASD")
else:
    print(f"{predictions} tidak menunukkan gejala ASD")

# Memasukkan prediksi ASD ke dalam frame yang sama. Untuk selanjutnya masuk ke rekomendasi terapi
user_input["ASD_traits"] = predicted_labels
user_input = user_input[["Speech", "Social", "Sensory", "Physical", "Total", "ASD_traits"]]

user_input_norm = (user_input - x_train.mean()) / x_train.std()
user_input_norm = user_input.to_numpy()

predictions = model2.predict(user_input_norm.reshape(1, -1))

# Mengambil 5 hasil prediksi tertinggi
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

# Menampilkan hasil prediksi dalam bentuk label asli
print(f"Top {top_n} Predicted Therapies (Original Labels):")
for label, prob in zip(top_indices, top_probabilities):
    original_label = label_asli[label]
    print(f"{original_label}: {prob:.4f}")


# Save model
model1.save("Model1.h5")
model2.save("Model2.h5")

# Konversi kedua model ke dalam format TFLite (.tflite)
converter1 = tf.lite.TFLiteConverter.from_keras_model(model1)
tflite_m1 = converter1.convert()

converter2 = tf.lite.TFLiteConverter.from_keras_model(model2)
tflite_m2 = converter2.convert()

combine_converter = tflite_m1 + tflite_m2
# Simpan kedua model TFLite ke dalam file yang sama
with open("Model.tflite", "wb") as f:
    f.write(combine_converter)
