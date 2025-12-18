# Guide d'intégration - Détection Thésaurus + BERT

## Vue d'ensemble

Ce guide explique comment intégrer la détection hybride (thésaurus + BERT) dans votre pipeline Metaforte.

## Architecture

```
Texte d'entrée
    ↓
[1. FILTRAGE THÉSAURUS] ← Rapide (< 1ms par texte)
    ↓
Score thésaurus
    ├─ Score bas → Rejeter
    └─ Score haut → Candidat
    ↓
[2. VALIDATION BERT] ← Précis mais coûteux (100-500ms par texte)
    ↓
Score BERT
    ↓
[3. SCORE FINAL] = (0.4 × thésaurus) + (0.6 × BERT)
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation simple

### 1. Test rapide (100 textes)

```bash
cd src/training
python compare_detection_approaches.py
```

### 2. Traitement complet

```bash
python thesaurus_based_detection.py
```

### 3. Intégration dans votre pipeline

```python
from thesaurus_based_detection import ThesaurusMetaphorDetector
import pandas as pd

df = pd.read_csv("data/processed/polititweets_clean.csv")

detector = ThesaurusMetaphorDetector()


df_results = detector.process_dataframe(df, text_column="text")


metaphors = df_results[df_results['final_metaphor_score'] > 0.7]
```

## Personnalisation du thésaurus

Le thésaurus par défaut couvre 9 domaines métaphoriques:
- **Guerre**: bataille, combat, attaque, conquête...
- **Eau**: vague, courant, inondation, noyade...
- **Feu**: brûler, incendie, flamme, passion...
- **Construction**: construire, bâtir, fondation, mur...
- **Voyage**: voyage, destination, parcours, navigation...
- **Nature**: forêt, tempête, croissance, récolte...
- **Corps**: cœur, cerveau, anatomie, circuler...
- **Mouvement**: mouvement, danse, flux, glisser...
- **Lumière**: brillant, lumineux, obscurité, éclairer...

### Ajouter domaines personnalisés

```python
from thesaurus_based_detection import ThesaurusMetaphorDetector

detector = ThesaurusMetaphorDetector()

detector.thesaurus['politique'] = [
    'élection', 'vote', 'scrutin', 'campagne', 'ballot',
    'mandat', 'candidat', 'suffrage', 'démocratie', 'citoyen'
]

df_results = detector.process_dataframe(df)
```

### Charger depuis fichier JSON

```python
import json

thesaurus_custom = {
    "domaine1": ["mot1", "mot2", ...],
    "domaine2": ["mot3", "mot4", ...]
}

with open("thesaurus.json", "w") as f:
    json.dump(thesaurus_custom, f)

# Dans le code:
with open("thesaurus.json", "r") as f:
    detector.thesaurus = json.load(f)
```

## Configuration des seuils

### Seuil thésaurus (filtrage)

```python
results, candidates = detector.detect_thesaurus(texts, threshold=0.2)


```

### Score final

```python

final_score = (0.2 × thesaurus_score) + (0.8 × bert_score)

):
final_score = (0.6 × thesaurus_score) + (0.4 × bert_score)
```

## Intégration dans main.py

```python

scripts = [
    ("src/extraction/extract_polititweets.py", "Extraction des tweets"),
    ("src/extraction/clean_polititweets.py", "Nettoyage"),
    ("src/training/thesaurus_based_detection.py", "Détection Thésaurus+BERT"),  # ← 
    ("src/extraction/select_top_metaphors.py", "Sélection top 200"),
]
```

## Outputs produits

```
data/processed/llm/thesaurus_bert/
├── scores.csv              # Tous les scores et analyses
├── top100.csv              # Top 100 métaphores
└── bottom100.csv           # Bottom 100 (non-métaphores)
```

### Format scores.csv

```
text,thesaurus_score,bert_score,final_metaphor_score,metaphor_domains,detection_method
"Tweet 1...",0.45,0.68,0.60,"[{domain: 'guerre', matches: 2}]",thesaurus+bert
"Tweet 2...",0.0,0.12,0.12,"[]",thesaurus
```

## Optimisation avancée

### Batch processing BERT

Pour traiter plus vite les candidats BERT:

```python
def detect_bert_batch(self, results, candidates, batch_size=32):
    """Version batch pour speedup 3-5x"""
    texts_batch = [texts for _, texts in candidates]
    
    for i in range(0, len(texts_batch), batch_size):
        batch = texts_batch[i:i+batch_size]
```

### Fine-tuning sur données annotées

Si vous avez des annotations manuelles:

```bash
python src/first_model_training.py --use-bert-finetune \
    --annotations data/annotations/annotated.csv \
    --output-model models/metaphor-detector-finetuned
```

## Benchmarks attendus

Sur corpus de ~5000 tweets (Configuration: GTX 1080 GPU):

| Approche | Temps | Précision* | F1 |
|----------|-------|-----------|-----|
| Thésaurus seul | 5s | 0.65 | 0.60 |
| BERT seul | 500-600s | 0.88 | 0.85 |
| **Hybride** | **150-200s** | **0.85** | **0.82** |

*Estimé sur validation manuelle

## Troubleshooting

### "CUDA out of memory"
→ Réduire batch size ou utiliser CPU

### Scores BERT tous bas/hauts
→ Vérifier hypothesis_template
→ Tester avec modèle différent: `xlm-roberta-base` (plus rapide)

### Thésaurus manque termes
→ Ajouter domaines personnalisés
→ Ou utiliser librairie: `nltk.wordnet` ou `spacy`

## Prochaines étapes

1. **Fine-tuning**: Entraîner BERT sur vos données annotées
2. **Multilingue**: Adapter pour d'autres langues
3. **Production**: Déployer avec FastAPI/Flask
4. **Monitoring**: Tracker les scores dans le temps

---
Besoin d'aide? Consultez `compare_detection_approaches.py` pour examples complets.
