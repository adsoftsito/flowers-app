# TensorFlow and tf.keras
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras


DATA_DIR = "dataset"
IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]


def preprocess_image(image_path):
    """Load and resize an image to the model's input shape."""
    image = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    return tf.image.resize(image, IMAGE_SIZE)


def build_model(num_classes):
    data_augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
            data_augmentation,
            keras.layers.Rescaling(1.0 / 255),
            keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_datasets(data_dir=DATA_DIR):
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def train(data_dir=DATA_DIR, epochs=30):
    train_ds, val_ds = load_datasets(data_dir)
    class_names = train_ds.class_names
    model = build_model(len(class_names))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    model.save("flowers_classifier.keras")
    export_path = "./flowers-classifier/1"
    model.export(export_path)
    print("Class names:", class_names)
    return model, class_names


def predict(model, image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(tf.expand_dims(image, axis=0), verbose=0)
    class_index = int(np.argmax(prediction[0]))
    return CLASS_NAMES[class_index], float(np.max(prediction[0]))


def main():
    train()


if __name__ == "__main__":
    main()
