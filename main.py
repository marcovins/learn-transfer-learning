from scripts.FilesProcessor import FilesProcessor
from models.Network_TF import Network_TF
from utils.imports import logger, os, argparse, Path

def contains_images(directory: Path):
    if not directory.exists():
        return False
    return any(file.suffix.lower() in ['.jpg', '.jpeg', '.png'] for file in directory.glob("*"))

def main(args):
    
    record_files = [
        os.path.join('data', 'train.tfrecord'),
        os.path.join('data', 'val.tfrecord'),
        os.path.join('data', 'test.tfrecord'),
        os.path.join('data', 'label_map.json')
    ]

    if not all(os.path.exists(path) for path in record_files):
        logger.info("Arquivos TFRecord não encontrados. Gerando dados a partir do diretório raw...")
        files = FilesProcessor(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test
        )
        files.process()
    else:
        logger.warning("Dados já tratados. Etapa de preparação de dados ignorada.")
    
    logger.info("Etapa 2: Inicializando o modelo de Transfer Learning...")
    model = Network_TF(
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        base_dir=args.output_dir
    )

    logger.info("Etapa 3: Treinando o modelo...")
    history = model.fit()

    logger.info("Pipeline finalizado com sucesso!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento com Transfer Learning")

    # === Argumentos para processamento de arquivos ===
    parser.add_argument("--source_dir", type=str, default="data/raw", help="Diretório com imagens brutas")
    parser.add_argument("--output_dir", type=str, default="data", help="Diretório de saída com pastas train/val/test")
    parser.add_argument("--train", type=float, default=0.7, help="Proporção de treino")
    parser.add_argument("--val", type=float, default=0.2, help="Proporção de validação")
    parser.add_argument("--test", type=float, default=0.1, help="Proporção de teste")

    # === Argumentos para o modelo ===
    parser.add_argument("--img_size", type=int, default=224, help="Tamanho da imagem (altura e largura)")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamanho do batch")
    parser.add_argument("--epochs", type=int, default=20, help="Número de épocas de treino")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()
    main(args)
