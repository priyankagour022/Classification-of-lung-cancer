import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SpecificityAtSensitivity

# Set the path to your dataset
train_path = '/content/drive/MyDrive/lung/train'
test_path = '/content/drive/MyDrive/lung/test'
valid_path = '/content/drive/MyDrive/lung/valid'
image_shape = (446, 446, 3)
N_CLASSES = 2
BATCH_SIZE = 16

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    batch_size=BATCH_SIZE,
    target_size=(446, 446),
    class_mode='categorical',
    subset='training',
)

valid_generator = train_datagen.flow_from_directory(
    train_path,
    batch_size=BATCH_SIZE,
    target_size=(446, 446),
    class_mode='categorical',
    subset='validation',
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_path,
    batch_size=BATCH_SIZE,
    target_size=(446, 446),
    class_mode='categorical',
)

# Load the pre-trained DenseNet201 model without the top classification layer
dense_model = DenseNet201(include_top=False, pooling='max', weights='imagenet', input_shape=(image_shape))
for layer in dense_model.layers:
    if 'conv5' not in layer.name:
        layer.trainable = False

# Building Model
model = Sequential()
model.add(dense_model)
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))

model.summary()

METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    SpecificityAtSensitivity(sensitivity=0.5, name='specificity')
]

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=METRICS
)

history = model.fit(train_generator, epochs=100, validation_data=valid_generator)

model.evaluate(test_generator)
