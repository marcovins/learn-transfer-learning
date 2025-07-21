import os, shutil, random, logging, os, colorlog, argparse, gc, json, math
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm

# Configuração do handler e do formatter colorido
log_format = (
    "%(log_color)s[%(asctime)s] %(levelname)s: %(message)s (%(filename)s:%(lineno)d)"
)

log_colors = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red',
}

formatter = colorlog.ColoredFormatter(log_format, log_colors=log_colors)

handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Configuração do logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)