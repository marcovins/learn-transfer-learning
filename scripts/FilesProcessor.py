from utils.imports import os, shutil, random, Path, Image, UnidentifiedImageError, logger, tqdm, tf, json

class FilesProcessor:
    def __init__(self, source_dir="data/raw", output_dir="data", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, image_size=(224, 224)):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios devem somar 1.0"
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.ratios = {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio
        }
        self.image_size = image_size
        self.splits = {"train": [], "val": [], "test": []}
        self.label_map = {}

    def _is_valid_image(self, filepath):
        try:
            with Image.open(filepath) as img:
                img.verify()
            return True
        except (UnidentifiedImageError, OSError):
            return False

    def _gather_all_images(self, class_dir):
        return [f for f in class_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png'] and self._is_valid_image(f)]

    def _split_images(self, images):
        random.shuffle(images)
        total = len(images)
        n_train = int(total * self.ratios["train"])
        n_val = int(total * self.ratios["val"])
        return {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _process_class(self, class_name, class_index):
        src_dir = self.source_dir / class_name
        if not src_dir.is_dir():
            return

        images = self._gather_all_images(src_dir)
        if not images:
            logger.warning(f"Nenhuma imagem válida encontrada em {src_dir}")
            return

        logger.info(f"Classe '{class_name}' contém {len(images)} imagens válidas.")

        split_images = self._split_images(images)

        for split, files in split_images.items():
            for img_path in files:
                self.splits[split].append((img_path, class_index))

    def _write_tfrecord(self, split, image_label_pairs):
        output_path = self.output_dir / f"{split}.tfrecord"
        with tf.io.TFRecordWriter(str(output_path)) as writer:
            for img_path, label in tqdm(image_label_pairs, desc=f"Escrevendo {split}.tfrecord"):
                image = tf.io.read_file(str(img_path))
                try:
                    image_decoded = tf.image.decode_image(image, channels=3)
                    image_resized = tf.image.resize(image_decoded, self.image_size)
                    image_uint8 = tf.cast(image_resized, tf.uint8)
                    encoded = tf.io.encode_jpeg(image_uint8).numpy()
                except Exception as e:
                    logger.warning(f"Erro ao processar {img_path}: {e}")
                    continue

                feature = {
                    'image': self._bytes_feature(encoded),
                    'label': self._int64_feature(label),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

    def process(self):
        logger.info("Iniciando conversão para TFRecord...")

        class_names = [d.name for d in self.source_dir.iterdir() if d.is_dir()]
        self.label_map = {name: idx for idx, name in enumerate(sorted(class_names))}

        for class_name in class_names:
            self._process_class(class_name, self.label_map[class_name])

        for split in ['train', 'val', 'test']:
            self._write_tfrecord(split, self.splits[split])

        # ✅ Salvar o mapa de classes em JSON
        label_map_path = self.output_dir / "label_map.json"
        with open(label_map_path, "w") as f:
            json.dump(self.label_map, f)
        logger.info(f"Mapa de classes salvo em {label_map_path}")

        logger.info("Conversão finalizada com sucesso.")
        logger.info(f"Mapa de classes: {self.label_map}")

        sizes = {split: len(images) for split, images in self.splits.items()}
        sizes_path = self.output_dir / "dataset_sizes.json"
        with open(sizes_path, "w") as f:
            json.dump(sizes, f)
        logger.info(f"Tamanhos dos datasets salvos em {sizes_path}")


if __name__ == '__main__':
    processor = FilesProcessor()
    processor.process()
