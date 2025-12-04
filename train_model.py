"""
train_model.py
Treina um classificador de flores usando o tutorial do TensorFlow
e salva o modelo em disco (flower_model) + o arquivo class_names.json.
"""

import pathlib
import json
import tensorflow as tf

# Tamanho das imagens (padrão do tutorial)
IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32
EPOCHS = 10  # você pode aumentar depois se quiser

def baixar_e_carregar_dataset():
    """
    Baixa o dataset de flores e cria os datasets de treino e validação.
    Retorna: (train_ds, val_ds, class_names)
    """
    print("Baixando dataset de flores...")
    data_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

    data_root = tf.keras.utils.get_file(
        'flower_photos',
        origin=data_url,
        untar=True
    )
    data_dir = pathlib.Path(data_root) / "flower_photos"

    print(f"Dataset baixado em: {data_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print("Classes encontradas:", class_names)

    # Otimizações de performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

def criar_modelo(num_classes: int) -> tf.keras.Model:
    """
    Cria o modelo de rede neural para classificação de flores.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)  # logits
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.summary()
    return model

def treinar_e_salvar_modelo():
    """
    Faz todo o fluxo:
    - baixa/carrega dataset
    - cria modelo
    - treina
    - salva o modelo e as classes em disco
    """
    # 1. Dataset
    train_ds, val_ds, class_names = baixar_e_carregar_dataset()

    # 2. Modelo
    num_classes = len(class_names)
    model = criar_modelo(num_classes)

    # 3. Treino
    print("Iniciando treino...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # 4. Salvar modelo
    print("Salvando modelo em 'flower_model'...")
    model.export("flower_model")

    # 5. Salvar classes
    print("Salvando classes em 'class_names.json'...")
    with open("class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    print("Treino concluído com sucesso!")

if __name__ == "__main__":
    treinar_e_salvar_modelo()
