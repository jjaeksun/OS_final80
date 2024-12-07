from flask import Flask, render_template, request
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)

model = ViTForImageClassification.from_pretrained("nateraw/vit-age-classifier")
feature_extractor = ViTFeatureExtractor.from_pretrained("nateraw/vit-age-classifier")

@app.route("/")
def upload_file():
    return '''
    <!doctype html>
    <title>연령 예측하기</title>
    <h1>예측 할 이미지를 선택하세요</h1>
    <form method="POST" action="/predict" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''

@app.route("/predict", methods=["POST"])
def predict():
    try:
        
        file = request.files["file"]
        
       
        image = Image.open(file.stream)
    except UnidentifiedImageError:
       
        return "<h1>Error: Uploaded file is not a valid image. Please try again with a valid image file.</h1>", 400

   
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(-1).item()

    
    return f"<h1>예측되는 나이: {predicted_class}</h1>"

if __name__ == "__main__":
    app.run(debug=True)