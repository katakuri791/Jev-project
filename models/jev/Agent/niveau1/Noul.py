from langchain_typesafe import Noul, Score, TypeSafeClassifier
from dotenv import load_dotenv
load_dotenv()
classifier = TypeSafeClassifier()

QUESTIONS = {
    "grave_noul": Noul(instructions="Ce problème est-il grave ?"),
    "gravite_score": Score(
        instructions="Quelle est la gravité du problème ?",
        criteria=[
            "Cosmétique, aucun impact sur l'usage.",
            "Gêne partielle pour certains utilisateurs.",
            "Blocage total du service.",
        ],
    ),
}

cas = [
    "Le logo est légèrement décalé sur mobile.",               # faible
    "L'export PDF est lent, environ 30 secondes.",             # moyen
    "Certains utilisateurs ne reçoivent pas l'email de confirmation.",  # moyen
    "Plus personne ne peut se connecter.",                     # élevé
]

for c in cas:
    r = classifier.invoke({"state": c, "questions": QUESTIONS})
    s = r.scores["gravite_score"]
    print(f"{c[:50]:50} | noul={r.nouls['grave_noul'].noul:.2f} "
          f"| score={s.score:.2f} conf={s.confidence:.2f} | legend={s.legend}")