from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_typesafe import Choice, TypeSafeClassifier
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()

classifier = TypeSafeClassifier()
petit = ChatOpenAI(model="gpt-5.6-luna")
gros = ChatOpenAI(model="claude-opus-4-7")
SEUIL = 0.75

class Etat(TypedDict):
    question: str
    route: str
    confiance: float
    reponse: str

def classifier_noeud(etat: Etat) -> dict:
    r = classifier.invoke({
        "state": etat["question"],
        "questions": {"route": Choice(
            instructions="Quel modèle, le moins coûteux possible, peut répondre correctement ?",
            criteria={
                "petit": "Recherche directe, extraction, reformulation, question factuelle simple.",
                "gros": "Raisonnement en plusieurs étapes, mathématiques, code complexe, enjeu élevé.",
            },
        )},
    })
    c = r.choices["route"]
    return {"route": c.choice, "confiance": c.confidence}

def choisir_branche(etat: Etat) -> str:
    # En cas de doute, on choisit la sécurité : le gros modèle
    if etat["confiance"] < SEUIL:
        return "gros_modele"
    return "petit_modele" if etat["route"] == "petit" else "gros_modele"

def petit_noeud(etat): return {"reponse": petit.invoke(etat["question"]).content}
def gros_noeud(etat):  return {"reponse": gros.invoke(etat["question"]).content}

g = StateGraph(Etat)
g.add_node("classifier", classifier_noeud)
g.add_node("petit_modele", petit_noeud)
g.add_node("gros_modele", gros_noeud)
g.add_edge(START, "classifier")
g.add_conditional_edges("classifier", choisir_branche)
g.add_edge("petit_modele", END)
g.add_edge("gros_modele", END)
app = g.compile()

if __name__ == "__main__":
    question = input("Pose ta question : ")
    r = app.invoke({"question": question})
    modele = "petit modèle" if r["route"] == "petit" and r["confiance"] >= SEUIL else "gros modèle"
    print(f"\n[Jev] route={r['route']} confiance={r['confiance']:.2f} -> {modele}")
    print(f"\n{r['reponse']}")