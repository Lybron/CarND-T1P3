import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import os.path as path
import random

### Load Data ###
samples = []
totalCount = 0

def flip_image(image):
  """Flips images with OpenCV."""
  copy = np.copy(image)
  copy = np.fliplr(copy)
  return copy

def load_samples():
  """Loads recorded images from the project subfolder 'train_data'."""
    
  lines = []
  images = []
  measurements = []
  
  folders = [
             'tk1_run1',
             'tk1_run2',
             'tk1_run4',
             'tk2_run1',
             ]
    
  for folder in folders:
    with open('./train_data/{}/driving_log.csv'.format(folder)) as csvfile:
      reader = csv.reader(csvfile)
      """
      Keeps track of the number of images being used within each subfolder.
      This will be the number of images that meet the threshold requirements.
      """
     
      lineCount = 0

      for line in reader:
        lines.append(line)
 
    for line in lines:
      """
      Excludes images where there is a small steering angle.
      This is to prevent the model from having a bias toward straight driving.
      """
      thresh_angle = 0.0349066 # 2 degrees

      # Arbitrary correction factor
      correction = 0.3 # 17 degrees

      center_path = line[0]
      left_path = line[1]
      right_path = line[2]

      center_filename = center_path.split('/')[-1]
      center_path = './train_data/{}/IMG/'.format(folder) + center_filename

      left_filename = left_path.split('/')[-1]
      left_path = './train_data/{}/IMG/'.format(folder) + left_filename

      right_filename = right_path.split('/')[-1]
      right_path = './train_data/{}/IMG/'.format(folder) + right_filename

      center_angle = float(line[3])
   
      if path.isfile(center_path) and abs(float(line[3])) >= thresh_angle:
        samples.append((center_path, center_angle, False))
        samples.append((center_path, center_angle, True)) # Boolean determines whether to flip the image

        lineCount += 1

      if path.isfile(left_path) and abs(float(line[3])) >= thresh_angle:
        left_angle = center_angle + correction
        samples.append((left_path, left_angle, False))

      if path.isfile(right_path) and abs(float(line[3])) >= thresh_angle:
        right_angle = center_angle - correction
        samples.append((right_path, right_angle, False))

    global totalCount

    totalCount += lineCount
    print("Folder:",folder)
    print("{} images".format(lineCount))

load_samples()

print("Total original images:", totalCount)
print("Total augmesnted images: {}".format(len(samples)))
### End Load Data ###

### Generator ###
def generator(samples, batch_size=32):
  num_samples = len(samples)
  while 1:
    random.shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset: offset + int(batch_size)]
      images = []
      angles = []
      
      for batch_sample in batch_samples:
        img_path = batch_sample[0]
        angle = float(batch_sample[1])
        flipped = batch_sample[2]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if flipped:
          image = flip_image(image)
          angle = -angle
          
        assert image.shape == (160,320,3), "img %s has shape %r" % (img_path, image.shape)
        
        images.append(image)
        angles.append(angle)
      
      X_train = np.array(images)
      y_train = np.array(angles)
      
      assert len(images) == len(angles), "images (%d) != angles (%d)" % (len(images), len(angles))
      yield sklearn.utils.shuffle(X_train, y_train)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.3)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
### End Generator ###

### Model Architecture ###
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import time

NUM_EPOCHS = 3

model_name = 'model'

def train_model():
  start = time.time()
  
  def resize_img(input):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(input, (66, 200))
    
  # NVIDIA Architecture #
  model = Sequential()

  model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    
  model.add(Lambda(lambda x: x/255.0 - 0.5))
    
  model.add(Lambda(resize_img))
    
  model.add(Conv2D(24, (5, 5), activation="elu", strides=(2, 2)))
  model.add(Conv2D(36, (5, 5), activation="elu", strides=(2, 2)))
  model.add(Conv2D(48, (5, 5), activation="elu", strides=(2, 2)))
  model.add(Conv2D(64, (3, 3), activation="elu", strides=(2, 2)))
  model.add(Conv2D(64, (1, 1), activation="elu", strides=(2, 2)))
    
  model.add(Flatten())
  model.add(Dense(100))
  model.add(Dense(50))
  model.add(Dense(10))
  model.add(Dense(1))
    
  try:
    model.load_weights(model_name + '.h5')
    print('Loaded previously saved weights.')
    
  except:
    print('Error: could not load weights.')
    
    
  checkpoint = ModelCheckpoint(model_name + '.h5',
                                monitor='val_loss',
                                verbose=0,
                                save_best_only=True,
                                mode='auto')
      
  model.compile(loss='mse', optimizer='adam')
  history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                                        validation_data=validation_generator,
                                        nb_val_samples=len(validation_samples), nb_epoch=NUM_EPOCHS,
                                        callbacks=[checkpoint])
                                 
  end = time.time()
   
  print('Time to train:', round((end-start)/60, 3), 'minutes')
   
  # Summary of model architectue
  print(model.summary())

train_model()
### End Model Architecture ###
