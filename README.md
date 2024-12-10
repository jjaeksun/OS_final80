
# Age Prediction Flask App

<p style="font-size:18px; font-family:Arial;">This project is a simple Flask-based web application that predicts the age range of a person from an uploaded image using a pre-trained Vision Transformer (ViT) model.</p>

## Features

- **Image Upload**: Users can upload an image through the web interface.
- **Age Prediction**: The app predicts the approximate age range of the person in the image.
- **User-Friendly Interface**: Includes a minimal HTML form for uploading images.

## How It Works

1. **Pre-trained Model**:  
   <span style="font-size:16px;">The application uses the pre-trained `ViTForImageClassification` model from the Hugging Face Transformers library. The specific model is `nateraw/vit-age-classifier`.</span>

2. **Age Categories**:  
   <span style="font-size:16px;">The predicted age range is mapped to the following categories:</span>
   <pre>
   0: 0-10 years
   1: 11-20 years
   2: 21-30 years
   3: 31-40 years
   4: 41-50 years
   5: 51-60 years
   6: 61-70 years
   7: 71-80 years
   8: 81-90 years
   9: 91-100 years
   </pre>

3. **Flask Routes**:
   <ul style="font-size:16px;">
     <li><code>/</code>: Displays the upload form.</li>
     <li><code>/predict</code>: Handles the uploaded image, processes it, and predicts the age range.</li>
   </ul>

## Requirements

### Python Libraries
<p style="font-family:Courier New;">Install the necessary libraries by running:</p>
<pre>
pip install flask transformers pillow
</pre>

### Pre-trained Model
<p style="font-size:14px;">The application automatically downloads the required model and feature extractor (<code>nateraw/vit-age-classifier</code>) from Hugging Face when the app is first run.</p>

## Usage

1. Clone this repository:
   <pre>
   git clone <repository_url>
   cd <repository_folder>
   </pre>

2. Run the Flask application:
   <pre>
   python app.py
   </pre>

3. Open a browser and go to:
   <pre>
   http://127.0.0.1:5000/
   </pre>

4. Upload an image to predict the age range.

## Error Handling

<p style="font-size:14px;">If the uploaded file is not a valid image, the application displays an error message:</p>
<blockquote>"Error: Uploaded file is not a valid image. Please try again with a valid image file."</blockquote>

## Example

<p style="font-size:16px;">After uploading a valid image, the app will display a message like:</p>
<blockquote><code>Predicted Age: 21 - 30 years</code></blockquote>

## Future Enhancements

- Add support for additional image formats.
- Improve the user interface with CSS styling.
- Implement detailed error messages for common issues (e.g., unsupported file types).
