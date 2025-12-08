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
        ("src/training/zero_shot_metaphor_filter.py", "Détection de métaphores (modèle zero-shot)"),
        ("src/extraction/first_tri.py", "Création de l'échantillon pour annotation"),
        ("src/extraction/select_top_metaphors.py", "Sélection du top 200 métaphores"),
    ]
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
    
    # Résumé final
    print("\n" + "="*60)
    print(f" {success_count}/{len(scripts)} étapes complétées")
    print("="*60)
    
    if success_count == len(scripts):
        print("Pipeline exécuté avec succès")
        return 0
    else:
        print("Le pipeline s'est arrêté avec des erreurs")
        return 1

if __name__ == "__main__":
    sys.exit(main())
