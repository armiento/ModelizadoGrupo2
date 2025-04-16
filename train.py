import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Semilla reproducible
np.random.seed(2024)
tf.random.set_seed(2024)

WIDTH, HEIGHT, CHANNELS = 224, 224, 3

# -------------------------------
# ✅ Cargar datos Dog vs Cat
# -------------------------------
def load_dog_cat_data(train_dir, test_dir):
    if not os.path.exists(train_dir) or not os.listdir(train_dir):
        raise ValueError(f"El directorio de entrenamiento '{train_dir}' está vacío o no existe.")
    if not os.path.exists(test_dir) or not os.listdir(test_dir):
        raise ValueError(f"El directorio de prueba '{test_dir}' está vacío o no existe.")

    label_list = sorted(os.listdir(train_dir))
    output_n = len(label_list)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(WIDTH, HEIGHT),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(WIDTH, HEIGHT),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(WIDTH, HEIGHT),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    print(f"Clases detectadas: {label_list}")
    print(f"Número de clases: {output_n}")

    return train_generator, validation_generator, test_generator, label_list, output_n

def analyze_dataset_distribution(generator):
    class_counts = {class_name: 0 for class_name in generator.class_indices.keys()}
    for class_name, class_idx in generator.class_indices.items():
        class_path = os.path.join(generator.directory, class_name)
        if os.path.isdir(class_path):
            class_counts[class_name] = len(os.listdir(class_path))
    return pd.DataFrame.from_dict(class_counts, orient='index', columns=['Número de Imágenes']).sort_values('Número de Imágenes', ascending=False)

# -------------------------------
# ✅ Crear modelo
# -------------------------------
def create_dog_cat_classifier(output_n, learning_rate=1e-4, dropout_rate=0.3, n_dense=512):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(WIDTH, HEIGHT, CHANNELS))
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(n_dense, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(output_n, activation='softmax')
    ])

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            metrics.AUC(name='auc'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall')
        ]
    )

    return model

# -------------------------------
# ✅ Entrenar modelo
# -------------------------------
def train_model(model, train_generator, validation_generator, epochs=10):
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
    checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', save_best_only=True, verbose=1)

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )
    return history

# -------------------------------
# ✅ Evaluar modelo
# -------------------------------
def evaluate_model(model, test_generator, label_list):
    predictions = model.predict(test_generator)
    true_classes = test_generator.classes

    df = pd.DataFrame({
        'Etiqueta Verdadera': [label_list[idx] for idx in true_classes],
        'Predicción': [label_list[np.argmax(pred)] for pred in predictions],
        'Correcto': [label_list[true_classes[i]] == label_list[np.argmax(predictions[i])] for i in range(len(true_classes))]
    })

    acc_por_clase = df.groupby('Etiqueta Verdadera')['Correcto'].mean()
    return acc_por_clase

# -------------------------------
# ✅ Predecir imagen individual
# -------------------------------
def predict_dog_cat(model, image_path, label_list, plot=True):
    img = load_img(image_path, target_size=(WIDTH, HEIGHT))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)[0]
    top_indices = predictions.argsort()[-2:][::-1]
    top_labels = [label_list[idx] for idx in top_indices]
    top_probs = predictions[top_indices]

    pred_df = pd.DataFrame({
        'Clase': top_labels,
        'Probabilidad (%)': top_probs * 100
    })

    if plot:
        plt.figure(figsize=(8, 4))
        sns.barplot(x='Probabilidad (%)', y='Clase', data=pred_df)
        plt.title('Predicción Dog vs Cat')
        plt.tight_layout()
        plt.show()

    return pred_df

# -------------------------------
# ✅ Graficar entrenamiento
# -------------------------------
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()

    # Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# -------------------------------
# ✅ Main
# -------------------------------
def main():
    train_dir = 'dataset/test_set/test_set'
    test_dir = 'dataset/training_set/training_set'

    train_gen, val_gen, test_gen, label_list, output_n = load_dog_cat_data(train_dir, test_dir)
    print(analyze_dataset_distribution(train_gen))

    model = create_dog_cat_classifier(output_n)
    history = train_model(model, train_gen, val_gen)

    plot_training_history(history)

    # Cargar el mejor modelo guardado
    best_model = load_model("best_model.keras")

    print("\nAccuracy por clase:")
    print(evaluate_model(best_model, test_gen, label_list))

    # Guardar el modelo final (opcional)
    best_model.save("dog_cat_classifier_final.keras")

    # Ejemplo de predicción
    test_image = 'cat.jpg'
    result = predict_dog_cat(best_model, test_image, label_list)
    print(result)

if __name__ == "__main__":
    main()




