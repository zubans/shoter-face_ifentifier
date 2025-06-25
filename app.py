from flask import Flask, request, jsonify
import face_recognition
import uuid
import base64
import numpy as np
import os

app = Flask(__name__)
KNOWN_DIR = "known_faces"

os.makedirs(KNOWN_DIR, exist_ok=True)

def load_known_faces():
    faces = {}
    for fname in os.listdir(KNOWN_DIR):
        path = os.path.join(KNOWN_DIR, fname)
        encoding = np.load(path)
        player_id = os.path.splitext(fname)[0]
        faces[player_id] = encoding
    return faces

def save_new_face(player_id, encoding):
    path = os.path.join(KNOWN_DIR, f"{player_id}.npy")
    np.save(path, encoding)

@app.route("/recognize", methods=["POST"])
def recognize():
    data = request.get_json()
    img_b64 = data.get("image")
    if not img_b64:
        return jsonify({"error": "Missing image"}), 400

    try:
        img_data = base64.b64decode(img_b64)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = face_recognition.load_image_file(np_arr)
    except Exception as e:
        return jsonify({"error": "Invalid image", "detail": str(e)}), 400

    encodings = face_recognition.face_encodings(img)
    if not encodings:
        return jsonify({"error": "No face found"}), 400

    unknown = encodings[0]
    known_faces = load_known_faces()

    for pid, known in known_faces.items():
        match = face_recognition.compare_faces([known], unknown, tolerance=0.45)
        if match[0]:
            return jsonify({"playerId": pid, "isNew": False})

    new_id = str(uuid.uuid4())
    save_new_face(new_id, unknown)
    return jsonify({"playerId": new_id, "isNew": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
