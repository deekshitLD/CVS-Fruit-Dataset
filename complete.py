# 1. Define the problem
# ...

# 2. Collect and preprocess data
import tensorflow_datasets as tfds

train_data, validation_data = tfds.load('fruits_360', split=['train[:80%]', 'train[80%:]'], as_supervised=True)
test_data = tfds.load('fruits_360', split='test', as_supervised=True)

def preprocess_image(image, label):
  image = tf.image.resize(image, [224, 224])
  image = tf.cast(image, tf.float32) / 255.0
  return image, label

train_data = train_data.map(preprocess_image).shuffle(1000).batch(32)
validation_data = validation_data.map(preprocess_image).batch(32)
test_data = test_data.map(preprocess_image).batch(32)

# 3. Design and train the model
from tensorflow.keras import layers, models

model = models.Sequential([
  layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
  layers.MaxPooling2D((2, 2)),
  layers.Conv2D(64, (3, 3), activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.Conv2D(128, (3, 3), activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.Conv2D(128, (3, 3), activation='relu'),
  layers.MaxPooling2D((2, 2)),
  layers.Flatten(),
  layers.Dropout(0.5),
  layers.Dense(512, activation='relu'),
  layers.Dense(120, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, epochs=20, validation_data=validation_data)

# 4. Evaluate the model
test_loss, test_acc = model.evaluate
