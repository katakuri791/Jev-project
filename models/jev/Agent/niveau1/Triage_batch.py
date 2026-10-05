import csv
from langchain_typesafe import Choice, Noul, TypeSafeClassifier
from collections import Counter
from dotenv import load_dotenv
load_dotenv()
classifier = TypeSafeClassifier()



QUESTIONS = {
    "urgent": Noul(instructions="Cette demande doit-elle être traitée aujourd'hui ?"),
    "equipe": Choice(
        instructions="Quelle équipe doit prendre en charge cette demande ?",
        criteria={
            "facturation": "Paiements, factures, remboursements, abonnements.",
            "technique": "Bugs, pannes, connexion, erreurs de l'application.",
            "autre": "Tout ce qui ne relève d'aucune des deux équipes.",
        },
    ),
}

with open("tickets.csv") as f:
    lignes = list(csv.DictReader(f))

entrees = [{"state": l["texte"], "questions": QUESTIONS} for l in lignes]
reponses = classifier.batch(entrees)

confusion = Counter()
justes = 0
for l, r in zip(lignes, reponses):
    predit = r.choices["equipe"].choice
    confusion[(l["equipe_attendue"], predit)] += 1
    justes += predit == l["equipe_attendue"]

print(f"Précision : {justes}/{len(lignes)}")
for (attendu, predit), n in sorted(confusion.items()):
    print(f"attendu={attendu:12} prédit={predit:12} : {n}")