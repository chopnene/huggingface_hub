from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient

app = Flask(__name__)

# Replace with your model name
MODEL_NAME = "stabilityai/stable-diffusion-2"
client = InferenceClient(model=MODEL_NAME)

@app.route("/")
def home():
    return "Hugging Face Image Generator is running!"

@app.route("/generate", methods=["POST"])
def generate_image():
    data = request.get_json()
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        image = client.text_to_image(prompt, guidance_scale=7)
        image_path = "output.png"
        image.save(image_path)
        return jsonify({"status": "success", "message": f"Image generated from: '{prompt}'"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
