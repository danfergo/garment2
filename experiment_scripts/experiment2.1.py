from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adadelta

import numpy as np
import operator
import pandas as pd
import pickle

DATASET_PATH = '/home/danfergo/SIG/Code/Experiments/data/deepfashion'
HISTORY_PATH = '/home/danfergo/SIG/Code/Experiments/history/experiment_2.1'
I_SIZE = 150

train_datagen = ImageDataGenerator(
)

validation_datagen = ImageDataGenerator(
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH + '/train',
    target_size=(I_SIZE, I_SIZE),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    DATASET_PATH + '/validation',
    target_size=(I_SIZE, I_SIZE),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

pkl_file = open(HISTORY_PATH + '_config.pkl', 'rb')
config = pickle.load(pkl_file)
pkl_file.close()

model = Model.from_config(config)
optimizer = Adadelta(lr=0.5, rho=0.95, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# (attempts) to load weights from file
try:
    model.load_weights(HISTORY_PATH + '_weights.pkl')
except:
    pass

# train per se
history = model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=2,
    validation_data=validation_generator,
    validation_steps=400,
    verbose=1)

# save weights
model.save_weights(HISTORY_PATH + '_weights.pkl')

# (re)save and load history
import pickle

past_history = {}

try:
    pkl_file = open(HISTORY_PATH + '_history.pkl', 'rb')
    past_history = pickle.load(pkl_file)
    pkl_file.close()
except:
    pass

full_history = {}

for k in history.history:
    if k in past_history:
        full_history[k] = np.concatenate((past_history[k], history.history[k]), axis=0)
    else:
        full_history[k] = history.history[k]

pkl_file = open(HISTORY_PATH + '_history.pkl', 'wb')
pickle.dump(full_history, pkl_file)
pkl_file.close()
