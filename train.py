import matplotlib.pyplot as plt
from numba import cuda

# reset gpu device before importing tensorflow
device = cuda.get_current_device()
device.reset()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNet
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

pre_trained_model = MobileNet(include_top=False, input_shape=(128,128,3))

for layer in pre_trained_model.layers:
  layer.trainable = False

print('last layer output shape: ', pre_trained_model.output_shape)

x = pre_trained_model.output

x = layers.GlobalAveragePooling2D()(x)

x = layers.Flatten()(x)

x = layers.Dense(512)(x)

x = layers.LeakyReLU()(x)

x = layers.Dropout(0.2)(x)

x = layers.Dense(6, activation = 'softmax')(x)

model = Model(pre_trained_model.input, x)

train_datagen = ImageDataGenerator(
    rescale = 1./255.,
    horizontal_flip=True,
    shear_range=0.2,  
    zoom_range=0.2,
    validation_split=0.1)

train_generator=train_datagen.flow_from_directory(
   'seg_train/seg_train/', 
    batch_size=64,
    target_size=(128,128),
    class_mode='categorical',
    subset='training'
    )

validation_generator=train_datagen.flow_from_directory(
    'seg_train/seg_train', 
    batch_size=64,
    target_size=(128,128), 
    class_mode='categorical',
    subset='validation'
    )

test_datagen = ImageDataGenerator(rescale = 1./255.)

test_generator = test_datagen.flow_from_directory(
    'seg_test/seg_test', 
    target_size=(128,128),
    batch_size=64,
    class_mode='categorical'
    )
    
print(train_generator.class_indices)

reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='auto')

model.compile(
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
  loss='categorical_crossentropy',
  metrics=['accuracy']
  )

history = model.fit(
  train_generator,
  validation_data = test_generator,
  epochs = 20,
  callbacks = [reduce],
  verbose = 1
  )

model.save("mobilenet_128.h5")

print('Layers :', len(model.layers))

accuracy = model.evaluate(test_generator)
print('Accuracy of the model on the test set: ', accuracy[1])

plt.show()
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))

plt.plot(epochs,acc,'b',label='Training Accuracy')
plt.plot(epochs,val_acc,'r',label='Validation Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs,loss,'b',label='Training Loss')
plt.plot(epochs,val_loss,'r',label='Validation Loss')
plt.legend()
plt.show()