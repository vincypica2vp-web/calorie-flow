import os
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from google import genai
from google.genai import types
import json

app = Flask(__name__)
CORS(app)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

@app.route('/')
def home():
    return render_template('index.html')

def search_open_food_facts(query):
    try:
        url = f"https://world.openfoodfacts.org/cgi/search.pl?search_terms={query}&search_simple=1&json=1&page_size=1"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if data.get("products") and len(data["products"]) > 0:
            product = data["products"][0]
            nutriments = product.get("nutriments", {})
            
            kcal = nutriments.get("energy-kcal_100g", nutriments.get("energy-kcal", 200))
            pro = nutriments.get("proteins_100g", 10)
            carb = nutriments.get("carbohydrates_100g", 20)
            fat = nutriments.get("fat_100g", 5)
            
            return {
                "name": product.get("product_name", query),
                "weight": 100, # Default confezionato 100g
                "kcal": int(kcal),
                "pro": round(float(pro), 1),
                "carb": round(float(carb), 1),
                "fat": round(float(fat), 1),
                "breakdown": f"Trovato su Open Food Facts (valori per 100g di prodotto)"
            }
    except Exception as e:
        print(f"Errore Open Food Facts: {e}")
    return None

@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    data = request.json
    food_query = data.get('query', '').strip()
    
    if not food_query:
        return jsonify({"error": "Query vuota"}), 400

    result = search_open_food_facts(food_query)
    if result:
        return jsonify(result)

    try:
        prompt = f"""
        Analizza questo alimento o piatto descritto dall'utente: "{food_query}".
        Se l'utente non ha specificato i grammi, **stima tu stesso il peso medio realistico in grammi** di una porzione standard per questo piatto/alimento. Calcola i totali nutrizionali complessivi per quel peso stimato.
        Restituisci ESCLUSIVAMENTE un oggetto JSON valido (senza blocchi di codice markdown aggiuntivi) con questa struttura esatta:
        {{
          "name": "Nome del piatto pulito",
          "weight": [peso stimato in grammi, numero intero, es. 120 o 350],
          "kcal": [calorie totali per quel peso, numero intero],
          "pro": [proteine totali in grammi, float],
          "carb": [carboidrati totali in grammi, float],
          "fat": [grassi totali in grammi, float],
          "breakdown": "Descrizione degli ingredienti e spiegazione del peso stimato"
        }}
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:-3].strip()
            
        parsed_data = json.loads(raw_text)
        return jsonify(parsed_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "Nessuna immagine caricata"}), 400

    try:
        image_bytes = file.read()
        
        prompt = """
        Analizza la foto di questo cibo. Riconosci il piatto o gli alimenti presenti, **stima tu stesso il peso medio in grammi** della porzione visibile e calcola i valori nutrizionali totali per quel peso.
        Restituisci ESCLUSIVAMENTE un oggetto JSON valido (senza blocchi di codice markdown) con questa struttura esatta:
        {
          "name": "Nome del piatto riconosciuto",
          "weight": [peso stimato in grammi, numero intero],
          "kcal": [calorie totali, numero intero],
          "pro": [proteine totali in grammi, float],
          "carb": [carboidrati totali in grammi, float],
          "fat": [grassi totali in grammi, float],
          "breakdown": "Riconoscimento visivo e stima del peso della porzione"
        }
        """

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=file.content_type or "image/jpeg",
                ),
                prompt
            ],
        )

        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:-3].strip()

        parsed_data = json.loads(raw_text)
        return jsonify(parsed_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)