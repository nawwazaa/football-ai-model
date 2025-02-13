import os
import cloudinary
import cloudinary.uploader
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

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

# Load YOLO Model
def load_yolo_model(model_path):
    print(f"Loading YOLO model from {model_path}...")
    return YOLO(model_path)

model = load_yolo_model(MODEL_PATH)

# Process Video Function
def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_path = ANNOTATED_VIDEO

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_path

@app.route("/")
def index():
    return jsonify({"message": "YOLO API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No video file received"}), 400

    # Save video file
    video_file = request.files['file']
    video_file.save(INPUT_VIDEO)

    try:
        # Process the video
        annotated_video_path = process_video(INPUT_VIDEO, model)
        print("Annotated video created:", annotated_video_path)

        if not os.path.exists(annotated_video_path):
            return jsonify({"error": "Annotated file not found."}), 500

        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(annotated_video_path, resource_type="video")
        cloudinary_url = upload_result.get("public_id")

        if not cloudinary_url:
            return jsonify({"error": "Upload failed"}), 500

        return jsonify({"videoUrl": cloudinary_url})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
