#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42


# # Specify each path

# In[ ]:


dataset = 'model/keypoint_classifier/keypoint.csv'
model_save_path = 'model/keypoint_classifier/keypoint_classifier.keras'

tflite_save_path = 'model/keypoint_classifier/keypoint_classifier.tflite'


# # Set number of classes

# In[ ]:


NUM_CLASSES = 4


# # Dataset reading
# Ye wo data hai jo AI dekhega (Hath ke 21 points ke coordinates).

# In[ ]:


X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))


# # Dataset reading
# $X$ = Sawal (Hath ke joints ki position/coordinates).
# $y$ = Jawab (Us gesture ka naam ya ID).

# In[ ]:


y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))


# # Spliting the DATA

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)


# # Model building

# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(21 * 2, )), #Mere model mein ek baar mein 42 numbers (21 points ke x,y) ek line mein lag kar andar aayenge.
    tf.keras.layers.Dropout(0.2), # 20 percent dropout
    tf.keras.layers.Dense(20, activation='relu'), #20 neurons in the first hidden layer with ReLU activation
    tf.keras.layers.Dropout(0.4), # 40 percent dropout
    tf.keras.layers.Dense(10, activation='relu'), #10 neurons in the second hidden layer with ReLU activation
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax') #Output layer with softmax activation
])


# In[ ]:


model.summary()  # tf.keras.utils.plot_model(model, show_shapes=True)


# In[ ]:


# Model checkpoint callback
# To save weights after each epoch
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=False)
# Callback for early stopping
# 20 round tak koi sudhaar nahi hua to training rok dega
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)


# In[ ]:


# Model compilation
# Loss Function - Ki kitni badi glti hui hai (sparse_categorical_crossentropy) - kyunki humare labels integer form mein hain
#Optimizer - us galti ko theek kese karna hai (adam) - ek popular optimization algorithm
# Metrics - accuracy (hum model ki accuracy dekhna chahte hain)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# # Model training

# In[ ]:


model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[cp_callback, es_callback]
)


# In[ ]:


# Model evaluation
val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)


# In[ ]:


# Loading the saved model
model = tf.keras.models.load_model(model_save_path)


# In[ ]:


# Inference test
predict_result = model.predict(np.array([X_test[0]]), verbose=0)
print(np.squeeze(predict_result))
print(np.argmax(np.squeeze(predict_result)))


# # Confusion matrix

# In[ ]:


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, classification_report

# def print_confusion_matrix(y_true, y_pred, report=True):
#     labels = sorted(list(set(y_true)))
#     cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    
#     df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
 
#     fig, ax = plt.subplots(figsize=(7, 6))
#     sns.heatmap(df_cmx, annot=True, fmt='g' ,square=False)
#     ax.set_ylim(len(set(y_true)), 0)
#     plt.show()
    
#     if report:
#         print('Classification Report')
#         print(classification_report(y_test, y_pred))

# Y_pred = model.predict(X_test)
# y_pred = np.argmax(Y_pred, axis=1)

# print_confusion_matrix(y_test, y_pred)


# # Convert to model for Tensorflow-Lite

# In[ ]:


# Save as a model dedicated to inference
model.save(model_save_path, include_optimizer=False)


# In[ ]:


# Transform model (quantization)

#TFLite basically ek special format hai jo mobile processors ke liye optimize hota hai.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Quantization krta hai ye line
#Mtlb weights ko 8-bit integers mein convert krega from 32-bit float.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Ye line model ko quantize karke TFLite format mein convert kar degi.
tflite_quantized_model = converter.convert()

#Save the quantized model to a file
# open(tflite_save_path, 'wb').write(tflite_quantized_model)
with open(tflite_save_path, 'wb') as f:
    f.write(tflite_quantized_model)


# # Inference test

# In[ ]:


#TFLite model ko "chalaane" wala software engine.
interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
#Memory allocate krta hai ye line, basically model ke liye jagah banata hai.
interpreter.allocate_tensors()


# In[ ]:


# Get I / O tensor
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# In[ ]:


interpreter.set_tensor(input_details[0]['index'], np.array([X_test[0]]))


# In[ ]:


# Inference implementation
interpreter.invoke() # Model ko chala raha hai
tflite_results = interpreter.get_tensor(output_details[0]['index'])

# Inference results
print(np.squeeze(tflite_results))
print(np.argmax(np.squeeze(tflite_results)))

