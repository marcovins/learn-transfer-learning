from .imports import logger
import tensorflow as tf

# Lista as GPUs disponíveis
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    logger.info(f"GPU(s) detectada(s): {[gpu.name for gpu in gpus]}")
else:
    logger.info("Nenhuma GPU detectada pelo TensorFlow.")

# Verifica qual dispositivo o TensorFlow está usando para computação
logger.info("Dispositivos disponíveis no TensorFlow:")
logger.info(tf.config.experimental.list_physical_devices())

# Opcional: verificar se o TensorFlow está usando GPU na execução de operações
tf.debugging.set_log_device_placement(True)

# Teste simples de computação para forçar TensorFlow usar GPU (se disponível)
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
c = tf.matmul(a, b)

logger.info("Resultado da multiplicação de matrizes:")
logger.info(c)
