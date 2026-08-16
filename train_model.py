import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

IMG_SIZE   = (224, 224)   # VGG16 requires exactly 224x224 pixels
BATCH_SIZE = 16           # How many images to process at once (16 is safe for CPU)
EPOCHS     = 10           # How many times to loop through all images
DATASET    = 'dataset'    # Path to your dataset folder

# ── STEP 1: Load VGG16 (pre-trained, no top layer) 
base_model = VGG16(
    weights     = 'imagenet',  # Use Oxford's pre-trained weights
    include_top = False,        # Remove final classification layer
    input_shape = (224, 224, 3) # 224x224 image, 3 colour channels (RGB)
)
base_model.trainable = False    # Freeze base — do not change its weights

# ── STEP 2: Add our crack-detection layers on top 
x = base_model.output
x = Flatten()(x)                # Convert 3D feature map to 1D list
x = Dense(256, activation='relu')(x)  # 256-neuron learning layer
x = Dropout(0.5)(x)             # Randomly drop 50% during training (prevents overfitting)
output = Dense(1, activation='sigmoid')(x)  # Final output: 0.0=no crack, 1.0=crack

# ── STEP 3: Build and compile the full model 
model = Model(inputs=base_model.input, outputs=output)
model.compile(
    optimizer = 'adam',              # Adam is the best general-purpose optimiser
    loss      = 'binary_crossentropy', # Standard loss for yes/no classification
    metrics   = ['accuracy']          # Show accuracy % during training
)

 
# ── STEP 4: Set up image loading with augmentation
train_datagen = ImageDataGenerator(
    rescale          = 1./255,   # Scale pixel values from 0-255 to 0-1
    horizontal_flip  = True,     # Randomly mirror images left-right
    rotation_range   = 10,       # Randomly rotate up to 10 degrees
    brightness_range = [0.8, 1.2] # Randomly vary brightness
)
val_datagen = ImageDataGenerator(rescale=1./255)  # No augmentation for validation

# ── STEP 5: Load images from your folders 
train_data = train_datagen.flow_from_directory(
    os.path.join(DATASET, 'train'),
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    class_mode  = 'binary'   # Two classes: crack / no_crack
)
 
val_data = val_datagen.flow_from_directory(
    os.path.join(DATASET, 'validation'),
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    class_mode  = 'binary'
)


# ── STEP 6: Print class mapping so you know which is which
print('Class indices:', train_data.class_indices)
# Expected output: {'crack': 0, 'no_crack': 1}  OR  {'crack': 1, 'no_crack': 0}
# WRITE THIS DOWN — you need it later in main.py

# ── STEP 7: Train the model 
print('Starting training... this will take 30-60 minutes on CPU.')
model.fit(
    train_data,
    validation_data = val_data,
    epochs          = EPOCHS,
    verbose         = 1   # Show progress bar
)
 
# ── STEP 8: Save the trained model 
os.makedirs('model', exist_ok=True)
model.save('model/crack_model.h5')
print('Model saved to model/crack_model.h5')
print('Training complete!')

loss, accuracy = model.evaluate(val_data)
print(f'Validation Loss:{loss:.4f}')
print(f'Validation Accuracy:{accuracy*100:;2f}%')