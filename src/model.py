import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf # type: ignore
def train_and_predict():
    data = np.array([
        [1, 60, 0],
        [2, 65, 0],
        [3, 70, 0],
        [4, 75, 1],
        [5, 80, 1],
        [6, 85, 1],
        [7, 90, 1],
        [8, 95, 1],
        [9, 97, 1],
        [10, 99, 1]
    ])
    X = data[:,:2]
    y = data[:,2]
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
    print(f"\nModel Accuracy:{accuracy*100:.2f}%")
    example = scaler.transform([[6.5,80]])
    prediction = model.predict(example)
    print("Predicted Pass Probability:", prediction[0][0])
    # print(x_train)
    # print(y_train)
    # print(y_test)
    # print(x_test)