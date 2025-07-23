
# 🚀 Projeto: Classificação de Imagens com Transfer Learning (TensorFlow) 🧠

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Este projeto implementa um pipeline completo de classificação de imagens utilizando Transfer Learning com TensorFlow/Keras. O objetivo é treinar um modelo para classificar imagens de diferentes classes (por exemplo: _🐱 gatos_ e _🐶 cachorros_) a partir de um conjunto de dados customizado, utilizando TFRecords para eficiência e reprodutibilidade.

## 📌 Sumário

- [🌐 Visão Geral](#🌐-visão-geral)
- [📂 Estrutura do Projeto](#📂-estrutura-do-projeto)
- [⚙️ Instalação](#⚙️-instalação)
- [🛠️ Pré-processamento dos Dados](#🛠️-pré-processamento-dos-dados)
- [🎓 Treinamento do Modelo](#🎓-treinamento-do-modelo)
- [⚡ Testando GPU](#⚡-testando-gpu)
- [📜 Arquivos Importantes](#📜-arquivos-importantes)
- [🤝 Como Contribuir](#🤝-como-contribuir)
- [📜 Licença](#📜-licença)

---

## 🌐 Visão Geral

O projeto utiliza a arquitetura **MobileNetV2** como base para Transfer Learning, adicionando uma _head_ personalizada para classificação. Os dados são organizados em **TFRecords** para maior performance. O treinamento é monitorado com callbacks, e os melhores pesos do modelo são salvos automaticamente com base na métrica de validação.

> **📌 Principais características**
> - ✅ Transfer Learning com MobileNetV2
> - ✅ Pipeline completo de pré-processamento com TFRecords
> - ✅ Salvamento automático dos melhores pesos
> - ✅ Suporte a GPU/TPU
> - ✅ Data Augmentation e callbacks inteligentes

## 📂 Estrutura do Projeto

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

## ⚙️ Instalação

1. Clone o repositório:
    ```sh
    git clone https://github.com/marcovins/learn-transfer-learning.git
    cd learn-transfer-learning
    ```

2. Crie um ambiente virtual (opcional, mas recomendado):
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # ou .venv\Scripts\activate no Windows
    ```

3. Instale as dependências:
    ```sh
    pip install -r requirements.txt
    ```

4. Organize seus dados em `data/raw/` conforme as classes (ex: `cats/`, `dogs/`).

## 🛠️ Pré-processamento dos Dados

Utilize o script `FilesProcessor.py` para gerar os arquivos TFRecord:

```sh
python scripts/FilesProcessor.py --input_dir data/raw --output_dir data
```

📌 **Saídas geradas:**
- `train.tfrecord`, `val.tfrecord`, `test.tfrecord`
- `label_map.json` — Mapeamento de classes para índices
- `dataset_sizes.json` — Tamanhos dos conjuntos de dados

## 🎓 Treinamento do Modelo

Exemplo de uso da classe principal:

```python
from models.Network_TF import Network_TF

net = Network_TF(
    image_size=(224, 224),
    batch_size=16,
    epochs=20,
    learning_rate=1e-4,
    base_dir='data'
)
history = net.fit()  # 🚀 Inicia o treinamento!
```

🔹 **Destaques do treinamento:**
- 📉 *EarlyStopping* — Interrompe automaticamente se não houver melhoria na validação.
- 💾 *ModelCheckpoint* — Salva automaticamente os melhores pesos do modelo.
- 📊 *TensorBoard* — Logs completos para visualização do desempenho.
- 🖼️ *Data Augmentation* — Aumento de dados em tempo real.
- 🔄 *Learning Rate Scheduler* — Ajuste dinâmico da taxa de aprendizado.
- 🏋️ *Gradient Clipping* — Evita explosão de gradientes.

## ⚡ Testando GPU

Para verificar a aceleração por GPU:

```sh
python utils/testGPU.py
```

O script [`utils/testGPU.py`](utils/testGPU.py) lista as GPUs disponíveis e realiza uma operação simples para garantir que a GPU está sendo usada.

## 📜 Arquivos Importantes

- [`models/Network_TF.py`](models/Network_TF.py): Implementação da arquitetura, treinamento e utilitários do modelo.
- [`utils/imports.py`](utils/imports.py): Importações centralizadas e configuração de logging.
- [`scripts/FilesProcessor.py`](scripts/FilesProcessor.py): Processamento dos dados e geração dos arquivos auxiliares.
- [`data/label_map.json`](data/label_map.json): Mapeamento de classes para índices.
- [`data/dataset_sizes.json`](data/dataset_sizes.json): Tamanhos dos conjuntos de dados.

## 🤝 Como Contribuir

1. Faça um fork do projeto.
2. Crie uma branch para sua feature (`git checkout -b minha-feature`).
3. Commit suas alterações (`git commit -am 'Adiciona nova feature'`).
4. Faça push para a branch (`git push origin minha-feature`).
5. Abra um Pull Request.

## 📜 Licença

Este projeto está licenciado sob a licença MIT.
