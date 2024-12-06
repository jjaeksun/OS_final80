from flask import Flask, render_template, request
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image

app = Flask(__name__)


model = ViTForImageClassification.from_pretrained("nateraw/vit-age-classifier")
feature_extractor = ViTFeatureExtractor.from_pretrained("nateraw/vit-age-classifier")

@app.route("/")
def upload_file():
    return '''
    <!doctype html>
    <title>Age Prediction</title>
    <h1>Upload an image to predict age</h1>
    <form method="POST" action="/predict" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    image = Image.open(file.stream)

    
    inputs = feature_extractor(images=image, return_tensors="pt")


    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(-1).item()

   
    return f"<h1>Predicted Age Group: {predicted_class}</h1>"

if __name__ == "__main__":
    app.run(debug=True)