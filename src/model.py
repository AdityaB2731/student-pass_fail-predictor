import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf # type: ignore
import joblib
import pandas as pd
def train_and_predict():
    data = pd.read_csv("student_pass_data.csv")

    X = data[['Age','Score']]
    y = data['Passed']

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)

    x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.2,random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8,activation="relu",input_shape=(2,)),
        tf.keras.layers.Dense(1,activation="sigmoid")
    ])

    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

    model.fit(x_train,y_train,epochs=100,verbose=1)

    loss,accuracy = model.evaluate(x_test,y_test,verbose=1)

    model.save("model.keras")  # instead of model.h5

    joblib.dump(scaler,"scaler.pkl")

    print(f"\nModel Accuracy:{accuracy*100:.2f}%")
    example = scaler.transform([[18, 75]])  # Age 18, Score 75
    prediction = model.predict(example)
    print("Predicted Pass Probability:", prediction[0][0])

    # print(x_train)
    # print(y_train)
    # print(y_test)
    # print(x_test)