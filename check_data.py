import json
import requests
from collections import Counter

DATA_URL = "https://raw.githubusercontent.com/Aatrox103/multilingual-llm-features/main/SAE/data/multilingual_data.jsonl"
response = requests.get(DATA_URL)
response.raise_for_status()

TARGET_LANGUAGES = ['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh', 'ar']

lines = response.text.strip().split('\n')
print(f"Total lignes : {len(lines)}\n")

print("=== 15 premières entrées ===")
for i, line in enumerate(lines[:15]):
    obj = json.loads(line)
    print(f"  [{i:3d}] {obj}")

print("\n=== Champs disponibles ===")
sample = json.loads(lines[0])
print(f"  Clés : {list(sample.keys())}")

if 'lan' in json.loads(lines[0]):
    print("\n=== Langue par bloc de 100 ===")
    for i, lan in enumerate(TARGET_LANGUAGES):
        block = lines[i*100 : i*100+100]
        langs_in_block = [json.loads(l).get('lan') for l in block]
        counts = Counter(langs_in_block)
        match = "Yes" if counts.most_common(1)[0][0] == lan else "No"
        print(f"  Bloc {i} (attendu: {lan}) → {dict(counts)} {match}")
else:
    print("\nPas de champ 'lang' détecté — vérification manuelle nécessaire")
    print("\n=== Textes par bloc de 100 (extrait) ===")
    for i, lan in enumerate(TARGET_LANGUAGES):
        sample_text = json.loads(lines[i*100])
        text = sample_text.get('text', sample_text)
        print(f"  Bloc {i} (attendu: {lan}) → \"{str(text)[:80]}\"")