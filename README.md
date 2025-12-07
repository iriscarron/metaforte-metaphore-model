# Metaforte - Modèle de détection de métaphores

Projet d'entraînement d'un modèle de détection de métaphores à partir du corpus Polititweets.

## Installation

```bash
pip install -r requirements.txt
```

## Extraction des données

```bash
cd src/extraction
python extract_polititweets.py
```

## Entraînement du modèle

```bash
cd src/training
python train_metaphor_model1.py
```

## Structure

- `data/raw/polititweet/` : Corpus TEI-XML
- `src/extraction/` : Scripts d'extraction
- `src/training/` : Scripts d'entraînement
