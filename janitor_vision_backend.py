
from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch

app = Flask(__name__)

# Carrega o modelo BLIP para legendas de imagem
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    image_url = data.get("image_url")
    if not image_url:
        return jsonify({"error": "Nenhuma URL de imagem fornecida."}), 400

    try:
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        output = model.generate(**inputs)
        description = processor.decode(output[0], skip_special_tokens=True)
        return jsonify({"description": description})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return "API de Visão (BLIP) funcionando!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
