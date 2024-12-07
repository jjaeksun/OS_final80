from flask import Flask, render_template, request
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)

model = ViTForImageClassification.from_pretrained("nateraw/vit-age-classifier")
feature_extractor = ViTFeatureExtractor.from_pretrained("nateraw/vit-age-classifier")

age_categories = {
    0: (0, 10),
    1: (11, 20),
    2: (21, 30),
    3: (31, 40),
    4: (41, 50),
    5: (51, 60),
    6: (61, 70),
    7: (71, 80),
    8: (81, 90),
    9: (91, 100)
}

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
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()

    predicted_age_range = age_categories.get(predicted_class, "Unknown Age Range")
    
    return f"<h1>예측되는 나이: {predicted_age_range[0]} - {predicted_age_range[1]}세</h1>"

if __name__ == "__main__":
    app.run(debug=True)
