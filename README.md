<div align="center">
  <h1>🐮 Cow Identification Challenge 📸</h1>
  <p><strong>Identificação Individual de Bovinos Leiteiros através de Visão Computacional e Biometria de Imagens</strong></p>
</div>

<br/>

Este repositório contém o código-fonte desenvolvido para resolver o desafio de **identificação não-invasiva de vacas**, utilizando **Computer Vision (YOLO)** para detecção de posturas anatômicas (Keypoints) e **Machine Learning Clássico (XGBoost, Random Forest, MLP, etc)** para classificação baseada em keypoints e padrões visuais da pelagem.

---

## 🎯 Objetivo do Projeto

O objetivo principal é extrair características biométricas de vacas capturadas por câmeras de Top-Down view, para conseguir identificar unicamente cada animal num rebanho leiteiro. 

Em vez de depender de métodos invasivos/estressantes como colares, brincos ou marcações a fogo, a biometria permite fazer a **rastreabilidade animal** e **pecuária de precisão** de maneira 100% digital, rápida e menos traumática.

---

## ✨ Principais Etapas do Pipeline

O fluxo de processamento foi subdividido em vários scripts sequenciais na pasta `notebooks/`, garantindo rastreabilidade e modularidade:

### 1. Preparação e Curadoria de Dados
- `00_validate_annotations.py` - Script para validação rigorosa da qualidade das anotações (bbox, recortes e keypoints).
- `01_organize_dataset.py` / `01a_rename_jsons.py` - Tratamento inicial e pareamento entre imagens raw e anotações JSON provenientes de plataformas de curadoria (Label Studio).
- `03_data_parsing.py` / `03a_offline_augmentation.py` - Scripts para formatar os dados no formato aceito pelo YOLO (TXT) e gerar novas imagens (Data Augmentation) introduzindo ruídos e generalização para o treinamento.

### 2. Visão Computacional (Keypoint Estimator)
- `04_train_keypoints.py` - Treinamento do modelo Deep Learning **YOLOn26-Pose** capaz de encontrar partes anatômicas fundamentais da vaca.

### 3. Extração de Features Geométricas e Imagem
- `05_evaluate_test_set.py` - Avaliação e inferência nas validações visuais do YOLO treinado.
- `06_feature_extraction.py` - **Fase crucial do pipeline**. Envia imagens para a rede YOLO e extrai a posição X e Y das partes da vaca. Depois, processa 2 opções estratégicas de modelagem biométrica:
  - **Estratégia 1 (S1):** Áreas métricas exclusivas da garupa (Rump Area).
  - **Estratégia 2/3 (S2):** Áreas globais (Dorsal + Garupa).
  - **Modo Extrator (geo/both):** Extração modular baseada unicamente nas distâncias *euclidianas* (geo), ou agregando características da cor da *pelagem e constraste* (both/híbrido).

### 4. Classificação ML
- `08_classification_id.py` - Script final de treinamento focado em Machine Learning para tabular as features extraídas anteriormente e prever a identidade visual correta do animal. Suporta modelos avançados, como XGBoost, LightGBM, CatBoost e MLP Neural Networks.

---

## 🚀 Como Começar

### 1. Clone o repositório
```bash
git clone https://github.com/rpa1tera/cow-challenge.git
cd cow-challenge
```

### 2. Crie e ative um ambiente virtual (Recomendado)
```bash
python -m venv .venv
# Ativar no Windows Powershell
.venv\Scripts\Activate.ps1   
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Extração e Treinamento
Para iniciar o pipeline de classificação, primeiro extraia as características biométricas rodando o script de extração. Você tem duas opções de modo (`--mode`):

```bash
# 1. Extrair APENAS features geométricas puras (distâncias):
python notebooks/06_feature_extraction.py --mode geo

# 2. Extrair features geométricas + padrão visual de pelagem (Biometria Híbrida):
python notebooks/06_feature_extraction.py --mode both
```

Com o arquivo CSV gerado (na pasta `data/processed/`), você pode treinar os modelos de classificação. O script permite combinar livremente o algoritmo de Machine Learning (`--model`), a estratégia de áreas do corpo (`--strategy 1, 2 ou 3`), e a fonte dos dados (`--mode geo ou both`).

**Exemplos de Treinamento:**

```bash
# Exemplo A: Treinar XGBoost usando apenas a área da Garupa (Estratégia 1) sem analisar pelos
python notebooks/08_classification_id.py --model xgboost --strategy 1 --mode geo

# Exemplo B: Treinar Random Forest usando Garupa + Dorso (Estratégia 3) sem analisar pelos
python notebooks/08_classification_id.py --model random_forest --strategy 3 --mode geo

# Exemplo C: Treinar Rede Neural (MLP) na área Dorsal (Estratégia 2) JUNTO com dados de pelagem
python notebooks/08_classification_id.py --model mlp --strategy 2 --mode both

# Exemplo D: Treinar LightGBM com TODAS as features (Geométricas S1+S2 + Pelagem Visual)
python notebooks/08_classification_id.py --model lightgbm --strategy 3 --mode both
```

---

## 🛠 Modelos Implementados no Projeto

* **Rede Extratora de Pose:** YOLO 26
* **Classificadores finais de Identidade:**
  * Multi-Layer Perceptron (MLP)
  * Random Forest
  * Support Vector Machine (SVM)
  * XGBoost
  * LightGBM
  * CatBoost

---

*Desenvolvido nas trincheiras da Pecuária de Precisão e IA de ponta para automatizar o campo.* 🌾🐄🚀
