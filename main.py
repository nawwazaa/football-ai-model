import os
import cloudinary
import cloudinary.uploader
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import gc
import traceback

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins="*")

# Cloudinary Configuration
cloudinary.config(
    cloud_name="dximtsuzo",
    api_key="556435271379951",
    api_secret="WL4A9wfF9pHNL2ItDjHcrWd6Mn0",
    secure=True
)

# Model Path (Ensure you update this path correctly)
MODEL_PATH = "best.pt"  # Ensure this file exists in your project folder
INPUT_VIDEO = "input_video.mp4"
ANNOTATED_VIDEO = "annotated_video.mp4"

# Force CPU mode and optimize settings
from ultralytics import settings
settings.update({'tensorrt': False, 'cuda': False})

# Load YOLO Model
def load_yolo_model(model_path):
    try:
        print(f"Loading YOLO model from {model_path}...")
        model = YOLO(model_path)
        # Verify model is working
        model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        raise

# Load the model
model = load_yolo_model(MODEL_PATH)

# Process Video Function
def process_video(video_path, model):
    try:
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        output_path = ANNOTATED_VIDEO

        # Optimize video capture
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frame_skip = 2  # Process every 3rd frame

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 640))

            # Run inference
            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        return output_path

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise

# Routes
@app.route("/")
def index():
    return jsonify({"message": "YOLO API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    print("Entered /predict endpoint")
    if 'file' not in request.files:
        return jsonify({"error": "No video file received"}), 400

    try:
        # Save video file
        video_file = request.files['file']
        video_file.save(INPUT_VIDEO)
        print("Input video saved successfully")

        # Process the video
        annotated_video_path = process_video(INPUT_VIDEO, model)
        print("Annotated video created:", annotated_video_path)

        if not os.path.exists(annotated_video_path):
            return jsonify({"error": "Annotated file not found."}), 500

        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(annotated_video_path, resource_type="video")
        cloudinary_url = upload_result.get("secure_url")
        print("Video uploaded to Cloudinary:", cloudinary_url)

        if not cloudinary_url:
            return jsonify({"error": "Cloudinary upload failed"}), 500

        # Clean up temporary files
        if os.path.exists(INPUT_VIDEO):
            os.remove(INPUT_VIDEO)
        if os.path.exists(ANNOTATED_VIDEO):
            os.remove(ANNOTATED_VIDEO)

        # Force garbage collection
        gc.collect()

        return jsonify({"videoUrl": cloudinary_url})

    except Exception as e:
        print(f"Error in /predict: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "error": "Video processing failed",
            "details": str(e)
        }), 500

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
