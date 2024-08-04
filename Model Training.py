import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from config import LABEL_DICT

data = np.load('data.npy')
target = np.load('target.npy')

model = Sequential()
model.add(Conv2D(200, (3, 3), input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(100, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(len(LABEL_DICT), activation='softmax'))  # Ensure number of output units matches number of classes

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2)
checkpoint = ModelCheckpoint('model.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
history = model.fit(train_data, train_target, epochs=20, callbacks=[checkpoint], validation_split=0.2)
print(model.evaluate(test_data, test_target))
