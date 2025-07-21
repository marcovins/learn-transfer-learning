from utils.imports import (
    EarlyStopping,
    ModelCheckpoint,
    Dense, GlobalAveragePooling2D,
    MobileNetV2, Adam, Model,
    os, logger, tf, gc, K, json, math
)

class Network_TF:
    def __init__(self, image_size: tuple = (224, 224), batch_size: int = 16, epochs: int = 20,
                 learning_rate: float = 1e-4, base_dir: str = 'data', num_classes: int = None):
        logger.info("Inicializando Network_TF...")
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.base_dir = base_dir
        self.AUTOTUNE = tf.data.AUTOTUNE
        self.dataset_sizes = self._load_dataset_sizes()

        # Tenta carregar mapeamento de classes, se não foi passado manualmente
        if num_classes is not None:
            self.num_classes = num_classes
        else:
            self.num_classes = self._load_class_count()

        self._config_callbacks()
        self._pre_process()
        self._base_model = self._get_weights()
        self.head = self._custom_head()
        self._compile_model()
        logger.info("Network_TF pronta para uso.")

    def _load_class_count(self):
        # Espera um arquivo JSON salvo pelo FilesProcessor, ex: {"cat": 0, "dog": 1}
        label_map_path = os.path.join(self.base_dir, "label_map.json")
        if not os.path.exists(label_map_path):
            raise FileNotFoundError("Arquivo label_map.json não encontrado.")
        with open(label_map_path, "r") as f:
            label_map = json.load(f)
        logger.info(f"Mapa de classes carregado: {label_map}")
        return len(label_map)

    def _config_callbacks(self):
        logger.info("Configurando callbacks...")
        self.callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint("models/weights/best_model.keras", save_best_only=True)
        ]

    def _parse_example(self, example_proto):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.io.decode_jpeg(parsed['image'], channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, self.image_size)
        label = tf.one_hot(parsed['label'], depth=self.num_classes)
        return image, label
    
    def _load_dataset_sizes(self):
        sizes_path = os.path.join(self.base_dir, "dataset_sizes.json")
        if not os.path.exists(sizes_path):
            raise FileNotFoundError("dataset_sizes.json não encontrado.")
        with open(sizes_path, "r") as f:
            sizes = json.load(f)
        return sizes

    def _load_dataset(self, tfrecord_path):
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        dataset = dataset.map(self._parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(1000)
        dataset = dataset.repeat()   # <-- aqui: repete infinitamente
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.AUTOTUNE)

        return dataset

    def _pre_process(self):
        logger.info("Carregando datasets a partir de arquivos TFRecord...")

        train_path = os.path.join(self.base_dir, "train.tfrecord")
        val_path = os.path.join(self.base_dir, "val.tfrecord")

        if not os.path.exists(train_path) or not os.path.exists(val_path):
            raise FileNotFoundError("Arquivos TFRecord não encontrados.")

        self._train_dataset = self._load_dataset(train_path)
        self._val_dataset = self._load_dataset(val_path)

    def _get_weights(self):
        logger.info("Carregando MobileNetV2 com pesos da ImageNet...")
        input_shape = self.image_size + (3,)
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
        base_model.trainable = False
        logger.info("Base congelada.")
        return base_model

    def _custom_head(self):
        logger.info("Construindo head personalizada...")
        x = self._base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        output = Dense(self.num_classes, activation='softmax')(x)
        logger.info("Head construída com sucesso.")
        return Model(inputs=self._base_model.input, outputs=output)

    def _compile_model(self):
        logger.info("Compilando modelo final...")
        self.head.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.info("Compilação concluída.")

    def fit(self):
        logger.info("Iniciando treinamento do modelo...")

        steps_per_epoch = math.ceil(self.dataset_sizes['train'] / self.batch_size)
        validation_steps = math.ceil(self.dataset_sizes['val'] / self.batch_size)

        history = self.head.fit(
            self._train_dataset,
            validation_data=self._val_dataset,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=self.callbacks
        )
        logger.info("Treinamento finalizado.")

        os.makedirs("models/weights", exist_ok=True)
        self.head.save("models/weights/final_model.keras")
        logger.info("Modelo salvo em models/weights/final_model.keras")

        self._clear_memory()
        return history

    def _clear_memory(self):
        logger.info("Liberando memória...")
        K.clear_session()
        gc.collect()
