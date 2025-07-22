# Projeto: Transfer Learning com TensorFlow

Este projeto implementa um pipeline completo de classificação de imagens utilizando Transfer Learning com TensorFlow/Keras. O objetivo é treinar um modelo para classificar imagens de diferentes classes (ex: gatos e cachorros) a partir de um conjunto de dados customizado, utilizando TFRecords para eficiência e reprodutibilidade.

## Sumário

- [Visão Geral](#visão-geral)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Pré-processamento dos Dados](#pré-processamento-dos-dados)
- [Treinamento do Modelo](#treinamento-do-modelo)
- [Testando GPU](#testando-gpu)
- [Arquivos Importantes](#arquivos-importantes)
- [Como Contribuir](#como-contribuir)
- [Licença](#licença)

---

## Visão Geral

O projeto utiliza a arquitetura MobileNetV2 como base para Transfer Learning, adicionando uma "head" personalizada para classificação. Os dados são organizados em TFRecords para maior performance. O treinamento é monitorado com callbacks e os melhores pesos são salvos automaticamente.

## Estrutura do Projeto

```
.
├── main.py
├── requirements.txt
├── data/
│   ├── dataset_sizes.json
│   ├── label_map.json
│   ├── train.tfrecord
│   ├── val.tfrecord
│   ├── test.tfrecord
│   └── raw/
│       ├── cats/
│       └── dogs/
├── models/
│   ├── Network_TF.py
│   └── weights/
│       ├── best_model.keras
│       └── final_model.keras
├── scripts/
│   └── FilesProcessor.py
└── utils/
    ├── imports.py
    └── testGPU.py
```

- **data/**: Dados brutos e processados, incluindo TFRecords e mapeamentos de classes.
- **models/**: Código do modelo e pesos treinados.
- **scripts/**: Scripts utilitários para processamento de arquivos.
- **utils/**: Utilitários e configuração de logging.

## Instalação

1. Clone o repositório:
    ```sh
    git clone https://github.com/marcovins/learn-transfer-learning.git
    cd learn-transfer-learning
    ```

2. Instale as dependências:
    ```sh
    pip install -r requirements.txt
    ```

3. Certifique-se de ter os dados organizados em `data/raw/` conforme as classes.

## Pré-processamento dos Dados

Utilize o script [`scripts/FilesProcessor.py`](scripts/FilesProcessor.py) para gerar os arquivos TFRecord e os arquivos auxiliares (`label_map.json`, `dataset_sizes.json`).

Exemplo de uso:
```sh
python scripts/FilesProcessor.py --input_dir data/raw --output_dir data
```

## Treinamento do Modelo

O treinamento é realizado pela classe [`models.Network_TF`](models/Network_TF.py):

```python
from models.Network_TF import Network_TF

net = Network_TF(
    image_size=(224, 224),
    batch_size=16,
    epochs=20,
    learning_rate=1e-4,
    base_dir='data'
)
history = net.fit()
```

- O modelo utiliza MobileNetV2 como base.
- Os pesos são salvos automaticamente em `models/weights/`.
- O treinamento utiliza callbacks de EarlyStopping e ModelCheckpoint.

## Testando GPU

Para verificar se o TensorFlow está utilizando GPU, execute:

```sh
python utils/testGPU.py
```

O script [`utils/testGPU.py`](utils/testGPU.py) lista as GPUs disponíveis e realiza uma operação simples para garantir que a GPU está sendo usada.

## Arquivos Importantes

- [`models/Network_TF.py`](models/Network_TF.py): Implementação da arquitetura, treinamento e utilitários do modelo.
- [`utils/imports.py`](utils/imports.py): Importações centralizadas e configuração de logging.
- [`scripts/FilesProcessor.py`](scripts/FilesProcessor.py): Processamento dos dados e geração dos arquivos auxiliares.
- [`data/label_map.json`](data/label_map.json): Mapeamento de classes para índices.
- [`data/dataset_sizes.json`](data/dataset_sizes.json): Tamanhos dos conjuntos de dados.

## Como Contribuir

1. Faça um fork do projeto.
2. Crie uma branch para sua feature (`git checkout -b minha-feature`).
3. Commit suas alterações (`git commit -am 'Adiciona nova feature'`).
4. Faça push para a branch (`git push origin minha-feature`).
5. Abra um Pull Request.

## Licença

Este projeto está licenciado sob a licença MIT.

---
