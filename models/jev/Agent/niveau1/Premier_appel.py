from langchain_typesafe import Choice, Noul, TypeSafeClassifier
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

messages = [
    # Cas clairs
    "J'ai été débité deux fois ce mois-ci, remboursez-moi.",
    "L'application plante dès que j'ouvre la page profil.",
    "Où se trouvent vos bureaux à Casablanca ?",
    "Le site est complètement en panne, nos clients ne peuvent plus payer !",
    "Je voudrais changer mon adresse de facturation.",
    "Impossible de me connecter depuis la mise à jour.",
    "Est-ce que vous recrutez des stagiaires ?",
    # Cas ambigus (volontairement)
    "Le paiement ne passe pas, j'ai une erreur 500.",   # facturation ou technique ?
    "Pas urgent, mais j'aimerais comprendre ma facture.",  # urgence faible explicite
    "Ça ne marche pas.",                                   # aucune information
]

for m in messages:
    r = classifier.invoke({"state": m, "questions": QUESTIONS})
    eq = r.choices["equipe"]
    print(f"{m[:45]:45} | urgent={r.nouls['urgent'].noul:.2f} "
          f"| {eq.choice} ({eq.confidence:.2f}) | {eq.probabilities}")