"""
Script principal pour exécuter le pipeline complet du projet Metaforte
"""
import subprocess
import sys
import os

def run_script(script_path, description):
    """Exécute un script Python et gère les erreurs"""
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print(f"{description} terminé avec succès")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de {description}")
        print(f"Code de sortie: {e.returncode}")
        print(f"Sortie standard: {e.stdout}")
        print(f"Sortie d'erreur: {e.stderr}")
        return False

def main():
    print("\n" + "="*60)
    print("PIPELINE METAFORTE - DÉTECTION DE MÉTAPHORES")
    print("="*60)
    
    scripts = [
        ("src/extraction/extract_polititweets.py", "Extraction des tweets depuis XML"),
        ("src/extraction/clean_polititweets.py", "Nettoyage et pré-tri des données"),
        ("src/training/zero_shot_metaphor_filter.py", "Détection de métaphores (XLM-RoBERTa zero-shot)"),
        ("src/training/zero_shot_metaphor_camembert.py", "Détection de métaphores (CamemBERT zero-shot)"),
        ("src/extraction/first_tri.py", "Création de l'échantillon pour annotation"),
        ("src/extraction/select_top_metaphors.py", "Sélection du top 200 métaphores"),
    ]

    annotations_path = "data/annotations/annotated.csv"
    eval_script = "src/visualization/eval_annotations.py"
    success_count = 0
    for script_path, description in scripts:
        if os.path.exists(script_path):
            if run_script(script_path, description):
                success_count += 1
            else:
                print(f"\nArrêt du pipeline  erreur ")
                break
        else:
            print(f"\n script introuvable: {script_path}")
            break

    # Évaluation si annotations disponibles
    if success_count == len(scripts):
        if os.path.exists(annotations_path):
            if os.path.exists(eval_script):
                if run_script(eval_script, "Évaluation (PR/F1) avec annotations"):
                    success_count += 1
            else:
                print(f"\n script introuvable: {eval_script}")
        else:
            print("\nAnnotations absentes (data/annotations/annotated.csv). Étape d'évaluation ignorée.")
    
    # Résumé final
    total_steps = len(scripts) + (1 if os.path.exists(annotations_path) else 0)

    print("\n" + "="*60)
    print(f" {success_count}/{total_steps} étapes complétées")
    print("="*60)
    
    if success_count == total_steps:
        print("Pipeline exécuté avec succès")
        return 0
    else:
        print("Le pipeline s'est arrêté avec des erreurs")
        return 1

if __name__ == "__main__":
    sys.exit(main())
