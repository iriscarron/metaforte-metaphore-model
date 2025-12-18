"""
Détection de métaphores basée sur thésaurus seul
Approche rapide et légère sans dépendances BERT
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent


class ThesaurusMetaphorDetector:
    """
    Détecteur de métaphores utilisant un thésaurus simple et efficace
    """
    
    def __init__(self):
        """Initialise le détecteur avec le thésaurus"""
        self.thesaurus = self._load_thesaurus()
    
    def _load_thesaurus(self) -> Dict[str, List[str]]:
        """
        Charge ou crée un thésaurus de mots métaphoriques
        Format: domaine -> liste de mots/expressions
        """
        thesaurus = {
            "guerre": [
                "bataille", "combat", "attaque", "défendre", "conquête",
                "armée", "ennemi", "victoire", "défaite", "stratégie",
                "offensive", "défensif", "soldat", "front", "arsenal",
                "arme", "tir", "blessure", "meurtre", "tuer"
            ],
            "eau": [
                "vague", "courant", "flot", "inondation", "submersion",
                "noyade", "flux", "reflux", "submerger", "déluge",
                "cascade", "torrent", "ruisseau", "océan", "mer",
                "tremper", "baigner", "flotter", "couler"
            ],
            "feu": [
                "brûler", "incendie", "flamme", "chaleur", "combustion",
                "étincelle", "allumer", "éteindre", "brasier", "embrasement",
                "inflammation", "ardeur", "passion", "enflammer", "consommer"
            ],
            "construction": [
                "construire", "bâtir", "fondation", "mur", "brique",
                "démolir", "structure", "édifice", "charpente", "ossature",
                "consolider", "architecture", "planifier", "édifier"
            ],
            "voyage": [
                "voyage", "destination", "chemin", "route", "parcours",
                "voyageur", "explorer", "découvrir", "errrance", "navigation",
                "embarquer", "départ", "arrivée", "étape", "trajet"
            ],
            "nature": [
                "forêt", "arbre", "plante", "fleur", "graine", "racine",
                "croissance", "cultiver", "récolte", "désert", "tempête",
                "ouragan", "tremblement", "éruption", "naturel"
            ],
            "corps": [
                "cœur", "cerveau", "tête", "bras", "jambe", "main",
                "corporel", "anatomie", "chair", "os", "sang", "circuler",
                "corps", "physique", "incarnation"
            ],
            "mouvement": [
                "mouvement", "immobile", "statique", "dynamique", "motion",
                "déplacement", "course", "marche", "danse", "flux",
                "circuler", "glisser", "danser", "tourner", "pivots"
            ],
            "lumière": [
                "lumière", "brillant", "lumineux", "obscurité", "ombre",
                "éclairer", "illuminer", "transparent", "opaque", "rayon",
                "lueur", "étincelle", "aube", "crépuscule", "éclat"
            ]
        }
        return thesaurus
    
    def _get_domain_score(self, text: str) -> Tuple[float, List[str]]:
        """
        Calcule un score basé sur les termes du thésaurus présents dans le texte
        
        Args:
            text: Texte à analyser
            
        Returns:
            Tuple (score, domaines_trouvés)
        """
        text_lower = text.lower()
        found_domains = []
        word_matches = []
        
        for domain, words in self.thesaurus.items():
            domain_matches = 0
            for word in words:
                if word.lower() in text_lower:
                    domain_matches += 1
                    word_matches.append((word, domain))
            
            if domain_matches > 0:
                found_domains.append({
                    'domain': domain,
                    'matches': domain_matches,
                    'words': [w for w, d in word_matches if d == domain]
                })
        
        # Score = nombre de domaines * poids + nombre de mots * poids
        if found_domains:
            score = (len(found_domains) * 0.3) + (sum(d['matches'] for d in found_domains) * 0.1)
            score = min(score, 1.0)  # Normaliser entre 0 et 1
            return score, found_domains
        
        return 0.0, []
    
    def detect(self, texts: List[str], threshold: float = 0.2) -> List[Dict]:
        """
        Détecte les métaphores basées sur thésaurus
        
        Args:
            texts: Liste de textes à analyser
            threshold: Seuil de score pour retenir un texte
            
        Returns:
            Liste de dictionnaires avec scores et domaines
        """
        results = []
        
        for text in texts:
            if not text.strip():
                results.append({
                    'text': text,
                    'metaphor_score': 0.0,
                    'domains': [],
                    'num_domains': 0,
                    'num_words': 0
                })
                continue
            
            score, domains = self._get_domain_score(text)
            
            result = {
                'text': text,
                'metaphor_score': score,
                'domains': domains,
                'num_domains': len(domains),
                'num_words': sum(d['matches'] for d in domains) if domains else 0
            }
            
            results.append(result)
        
        return results
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """
        Traite un DataFrame complet
        
        Args:
            df: DataFrame contenant les textes
            text_column: Nom de la colonne contenant les textes
            
        Returns:
            DataFrame augmenté avec colonnes de scores
        """
        texts = df[text_column].fillna("").tolist()
        
        print(f"Détection par thésaurus ({len(texts)} textes)...")
        results = self.detect(texts)
        
        # Ajouter les résultats au DataFrame
        df['metaphor_score'] = [r['metaphor_score'] for r in results]
        df['num_domains'] = [r['num_domains'] for r in results]
        df['num_words'] = [r['num_words'] for r in results]
        df['metaphor_domains'] = [json.dumps(r['domains']) for r in results]
        
        return df


def main():
    """Exécution principale"""
    input_path = PROJECT_ROOT / "data/processed/polititweets_clean.csv"
    df = pd.read_csv(input_path)
    
    # Traiter tous les textes
    detector = ThesaurusMetaphorDetector()
    df = detector.process_dataframe(df)
    
    # Sauvegarder les résultats
    output_dir = PROJECT_ROOT / "data/processed/llm/thesaurus"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "scores.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✓ Résultats sauvegardés: {output_path}")
    
    # Statistiques
    print("\n=== STATISTIQUES ===")
    print(f"Total textes: {len(df)}")
    print(f"Moyenne score: {df['metaphor_score'].mean():.3f}")
    print(f"Median score: {df['metaphor_score'].median():.3f}")
    print(f"Max score: {df['metaphor_score'].max():.3f}")
    print(f"Textes avec métaphore (score > 0.3): {(df['metaphor_score'] > 0.3).sum()}")
    
    # Top 100 et Bottom 100
    df_sorted = df.sort_values('metaphor_score', ascending=False)
    df_sorted.head(100)[['text', 'metaphor_score', 'num_domains', 'num_words']].to_csv(
        output_dir / "top100.csv", index=False, encoding='utf-8-sig'
    )
    df_sorted.tail(100)[['text', 'metaphor_score', 'num_domains', 'num_words']].to_csv(
        output_dir / "bottom100.csv", index=False, encoding='utf-8-sig'
    )
    
    print("✓ Top 100 et Bottom 100 générés")


if __name__ == "__main__":
    main()
