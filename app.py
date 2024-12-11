from flask import Flask, render_template, request
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)

# 모델 및 특성 추출기 로드
model = ViTForImageClassification.from_pretrained("nateraw/vit-age-classifier")
feature_extractor = ViTFeatureExtractor.from_pretrained("nateraw/vit-age-classifier")

# 연령 카테고리 매핑
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
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        image = Image.open(file.stream)
    except UnidentifiedImageError:
        return "<h1>유효하지 않은 이미지 파일입니다. 다시 업로드하세요.</h1>", 400

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()
    predicted_age_range = age_categories.get(predicted_class, "알 수 없는 나이 범위")
    predicted_age = f"{predicted_age_range[0]} - {predicted_age_range[1]}"

    return render_template("result.html", predicted_age=predicted_age)

if __name__ == "__main__":
    app.run(debug=True)
