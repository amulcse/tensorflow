import tensorflow as tf
import numpy as np

hw = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20];
squeareArea = [1,4,9,16,25,36,49,64,81,100,121,144,169,196,225,256,289,324,361,400];

layer0 = tf.keras.layers.Dense(units=4, input_shape=[1])
model = tf.keras.Sequential([layer0])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

model.fit(hw, squeareArea, epochs=500, verbose=False)

print("Finished training the model")

print(model.predict([25.0]))

print("Model predicts that 100 degrees Celsius is: {} degrees Fahrenheit".format(model.predict([25.0])))
print("These are the l0 variables: {}".format(layer0.get_weights()))


