import os
import json
import numpy as np
import google.generativeai as genai
import faiss
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
# MODIFICATION : Importation de la nouvelle bibliothèque de détection de langue
from langdetect import detect

# --- 1. Configuration et Chargement des Modèles et Données ---

print("🚀 Démarrage du serveur backend du chatbot (Mode Complet)...")

# Configuration de l'API Google Gemini
try:
    GOOGLE_API_KEY = "AIzaSyC_SP-vD9LQn69CN-PjLH0Vz4DlnSbc7Hw"  # Votre clé API
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini = genai.GenerativeModel("gemini-1.5-flash")
    print("✅ Clé API Google configurée et modèle 'gemini-1.5-flash' prêt.")
except Exception as e:
    print(f"❌ ERREUR CRITIQUE lors de la configuration de Gemini : {e}")
    gemini = None

# Chargement des modèles et des traducteurs
try:
    print("⏳ Chargement des modèles de langue et de traduction...")
    model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
    translator = GoogleTranslator(source="auto", target="fr")
    print("✅ Modèles de langue chargés.")
except Exception as e:
    print(f"❌ ERREUR lors du chargement des modèles SentenceTransformer : {e}")

# Chargement des données pour la recherche hybride
try:
    print("⏳ Chargement des données structurées et non structurées...")
    with open("structured_stats.json", "r", encoding="utf-8") as f:
        structured_stats = json.load(f)
    structured_embeddings = np.load("structured_embeddings.npy")
    print("   -> Données statistiques chargées.")
    faiss_index = faiss.read_index("data_index.faiss")
    with open("data_metadata.json", "r", encoding="utf-8") as f:
        text_metadata = json.load(f)
    print("   -> Index Faiss et métadonnées textuelles chargés.")
    print("✅ Toutes les données ont été chargées avec succès.")
except FileNotFoundError as e:
    print(f"❌ ERREUR CRITIQUE : Fichier de données manquant : {e}")
    structured_stats = []
    faiss_index = None


# --- 2. Fonctions de Recherche et de Génération (Logique complète) ---

def search_text(question, k=5):
    """
    Recherche dans les données textuelles non structurées à l'aide de Faiss.
    """
    if not faiss_index: return []
    print("   -> Traduction et recherche dans l'index de texte Faiss...")
    # On conserve la traduction en français car la base de connaissance est en français
    question_fr = translator.translate(question)
    q_emb = model.encode([question_fr]).astype("float32")
    distances, indices = faiss_index.search(q_emb, k)
    retrieved = [text_metadata[idx]["text"] for idx in indices[0]]
    print(f"   -> {len(retrieved)} morceaux de texte pertinents trouvés via Faiss.")
    return retrieved


def search_structured_stats_fast(question, top_k=5):
    """
    Recherche dans les données statistiques structurées.
    """
    if not structured_stats: return []
    print("   -> Traduction et recherche dans les données statistiques structurées...")
    # On conserve la traduction en français ici aussi
    question_fr = translator.translate(question)
    q_emb = model.encode([question_fr]).astype("float32")[0]
    sims = np.dot(structured_embeddings, q_emb) / (
                np.linalg.norm(structured_embeddings, axis=1) * np.linalg.norm(q_emb))
    top_indices = sims.argsort()[::-1][:top_k]
    retrieved = [structured_stats[i] for i in top_indices]
    print(f"   -> {len(retrieved)} lignes de données pertinentes trouvées.")
    return retrieved


# MODIFICATION : Remplacement de l'ancienne fonction de génération par la nouvelle fonction dynamique
def construire_prompt_dynamique(question, text_chunks, structured_rows, langue):
    """
    Construit le prompt pour le LLM en adaptant la langue des instructions.
    """
    # Formatage du contexte
    structured_context = "\n".join([
                                       f"- [{r.get('zone', 'N/A')} | {r.get('milieu', 'N/A')} | {r.get('sexe', 'N/A')}]: {r.get('indicator', 'N/A')} → {r.get('value', 'N/A')}"
                                       for r in structured_rows])
    text_context = "\n\n".join(text_chunks)

    # Instructions communes
    instructions_base = """
    Tu es un assistant conversationnel intelligent, naturel et serviable.
    Ton rôle principal est d'aider les utilisateurs en te basant sur des données du Haut-Commissariat au Plan (HCP).
    - SOIS NATUREL : Comporte-toi comme un humain. Si la question est une salutation simple comme "Bonjour" ou "Salam", réponds simplement.
    - SOIS DIRECT : Ne dis JAMAIS des phrases comme "Selon les documents fournis", "D'après le contexte", "Dans les informations que j'ai". Réponds directement à la question.
    - SOIS HONNÊTE : Si les informations ci-dessous ne te permettent pas de répondre, ou si elles sont vides, dis simplement que tu ne sais pas, de manière naturelle. Ne cherche pas dans tes connaissances générales, sauf pour les salutations.
    """

    # Instructions spécifiques à la langue
    if langue == 'fr':
        instructions_langue = """
        - RÉPONDS EN FRANÇAIS : La question est en français, ta réponse doit être en français clair et naturel.
        - EXEMPLE DE REFUS : Si tu ne sais pas, dis quelque chose comme "Je n'ai pas d'information précise à ce sujet." ou "Je ne saurais pas vous dire.".
        """
        question_label = "Question de l'utilisateur"
        reponse_label = "Ta réponse (en français) :"
    else:  # Par défaut, on considère que c'est de la Darija (langue 'ar' ou autre)
        instructions_langue = """
        - RÉPONDS EN DARIJA : La question est en Darija, ta réponse doit être en Darija marocaine claire et naturelle.
        - EXEMPLE DE REFUS : Si tu ne sais pas, dis quelque chose comme "ما عنديش شي معلومة مضبوطة على هاد الموضوع." ou "ما نقدرش نجاوبك على هادي.".
        """
        question_label = "السؤال ديال المستخدم"
        reponse_label = "الجواب ديالك (بالدارجة المغربية) :"

    # Assemblage final du prompt
    prompt_final = f"""
{instructions_base}
{instructions_langue}

---
CONTEXTE FOURNI (à utiliser si pertinent)
Contexte textuel (analyses générales) :
{text_context}

Contexte statistique (chiffres précis) :
{structured_context}
---

{question_label}:
{question}

{reponse_label}
"""
    print("\n--- PROMPT DYNAMIQUE ENVOYÉ À L'API GEMINI ---")
    print(prompt_final)
    print("-------------------------------------------------------\n")

    if not gemini:
        return "Service d'IA non configuré."

    try:
        print("   -> Envoi de la requête à l'API Gemini...")
        response = gemini.generate_content(prompt_final)
        print("   -> Réponse reçue de l'API.")
        return response.text
    except Exception as e:
        print(f"❌ ERREUR lors de l'appel à l'API Gemini : {e}")
        return "Désolé, une erreur technique est survenue lors de la communication avec l'IA."


# --- 3. API Flask ---

app = Flask(__name__)
CORS(app)


@app.route('/ask', methods=['POST'])
def ask_chatbot_api():
    if not request.json or 'question' not in request.json:
        return jsonify({'error': 'Requête invalide, "question" manquante.'}), 400

    question = request.json['question']
    print(f"\n\n=============================================")
    print(f"❓ Question reçue : {question}")

    # MODIFICATION : Détection de la langue de la question
    try:
        langue_detectee = detect(question)
        print(f"   -> Langue détectée : '{langue_detectee}'")
    except Exception as e:
        print(f"   -> Avertissement : Échec de la détection de langue ({e}). Utilisation du français par défaut.")
        langue_detectee = 'fr'

    # Pipeline de recherche hybride (cette partie ne change pas)
    text_chunks = search_text(question)
    structured_rows = search_structured_stats_fast(question)

    # MODIFICATION : Appel de la nouvelle fonction de génération de réponse
    answer = construire_prompt_dynamique(question, text_chunks, structured_rows, langue_detectee)

    print(f"🤖 Réponse finale générée : {answer}")
    return jsonify({'answer': answer})


@app.route('/', methods=['GET'])
def health_check():
    return "✅ Le serveur backend du chatbot (Mode Complet Hybride) est en marche."


# --- 4. Lancement du Serveur ---

if __name__ == '__main__':
    # MODIFICATION : Ajout de use_reloader=False pour la stabilité pendant le développement
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)  