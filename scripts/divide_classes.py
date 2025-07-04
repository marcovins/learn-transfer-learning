import os
import shutil
import random
from pathlib import Path
from PIL import Image, UnidentifiedImageError

# Caminhos
base_dir = Path("data/train")
val_dir = Path("data/val")
val_ratio = 0.2  # 20% para validação

def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()  # Verifica se é uma imagem válida
        return True
    except (UnidentifiedImageError, OSError):
        return False

# Garante que as pastas de validação existem
for class_name in os.listdir(base_dir):
    src_dir = base_dir / class_name
    dst_dir = val_dir / class_name

    if not src_dir.is_dir():
        continue  # pula arquivos ou outras pastas indevidas

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Lista de imagens válidas
    all_files = list(src_dir.glob("*"))
    images = [f for f in all_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png'] and is_valid_image(f)]

    random.shuffle(images)
    val_count = int(len(images) * val_ratio)
    val_images = images[:val_count]

    # Move imagens para pasta de validação
    for img_path in val_images:
        dst_path = dst_dir / img_path.name
        shutil.move(str(img_path), str(dst_path))
        print(f"Movido para validação: {dst_path}")
