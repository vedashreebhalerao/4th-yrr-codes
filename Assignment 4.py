# Import necessary libraries
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models, Sequential, layers, preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.applications.vgg16 import VGG16
import os
import mtcnn

# Load file names from the FaceRecog directory and create NameArray with labels
file_names = os.listdir("Assets/FaceRecog")
NameArray = []

# Categorize files based on name and append corresponding labels to NameArray
for name in file_names:
    category = name.split('.')[0]
    if category == 'gourishankar':
        NameArray.append('Gourishankar')
    elif category == 'Aditya_Panchwagh':
        NameArray.append("Aditya")
    elif category == "Dhananjay_Jha":
        NameArray.append("Dhananjay")
    elif category == 'Habil_Bhagat':
        NameArray.append("Habil")
    elif category == "Karan_Mahajan":
        NameArray.append("Karan")
    elif category == 'Kartik_Jawanjal':
        NameArray.append("Kartik")
    elif category == "Krish_Shah":
        NameArray.append("Krish")
    elif category == 'Manas_Oswal':
        NameArray.append("Manas")
    elif category == "Mayank_Modi":
        NameArray.append("Mayank")
    elif category == 'Shubham_Pagare':
        NameArray.append("Shubham")
    elif category == "Vishal_Kasa":
        NameArray.append("Kasa")

# Create a DataFrame to hold filenames and corresponding categories
train = pd.DataFrame({
    'filename': file_names,
    'category': NameArray
})

# Split data into training and validation sets
from sklearn.model_selection import train_test_split
train_df, validate_df = train_test_split(train, test_size=0.2, random_state=0)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

# Set up ImageDataGenerator for training and validation data augmentation
training_gen = ImageDataGenerator(
    rotation_range=5,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

trainingdata = training_gen.flow_from_dataframe(
    train_df,
    "Assets/FaceRecog",
    x_col='filename',
    y_col='category',
    target_size=(224, 224),
    class_mode='categorical'
)

validation_gen = ImageDataGenerator(
    rotation_range=5,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

validationdata = validation_gen.flow_from_dataframe(
    validate_df,
    "Assets/FaceRecog",
    x_col='filename',
    y_col='category',
    target_size=(224, 224),
    class_mode='categorical'
)

# Build the model using VGG16 as the base
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(400, activation='relu'),
    layers.Dense(6, activation='softmax')
])

# Compile the model
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(trainingdata, validation_data=validationdata, epochs=30)

# Evaluate the model on validation data
_, validation_acc = model.evaluate(validationdata, verbose=0)
print("Validation Accuracy:", validation_acc)
