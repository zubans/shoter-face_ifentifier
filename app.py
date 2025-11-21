from flask import Flask, request, jsonify
import face_recognition
import base64
import numpy as np
import os
import sys
import time
from io import BytesIO
from PIL import Image

app = Flask(__name__)
KNOWN_DIR = "known_faces"
APPEARANCE_DIR = "appearance_vectors"
APPEARANCE_THRESHOLD = 0.88

os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs(APPEARANCE_DIR, exist_ok=True)

import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')

_known_faces_cache = None
_known_faces_cache_time = 0
_appearance_cache = None
_appearance_cache_time = 0
CACHE_TTL = 60


def load_known_faces(use_cache=True):
    global _known_faces_cache, _known_faces_cache_time
    
    current_time = time.time()
    if use_cache and _known_faces_cache is not None and (current_time - _known_faces_cache_time) < CACHE_TTL:
        return _known_faces_cache
    
    faces = {}
    for fname in os.listdir(KNOWN_DIR):
        if not fname.endswith(".npy"):
            continue
        path = os.path.join(KNOWN_DIR, fname)

        try:
            encoding = np.load(path)
        except Exception:
            continue

        player_id = fname.split("_")[0]
        faces.setdefault(player_id, []).append(encoding)
    
    _known_faces_cache = faces
    _known_faces_cache_time = current_time
    return faces


def load_appearance_profiles(use_cache=True):
    global _appearance_cache, _appearance_cache_time

    current_time = time.time()
    if use_cache and _appearance_cache is not None and (current_time - _appearance_cache_time) < CACHE_TTL:
        return _appearance_cache

    profiles = {}
    for fname in os.listdir(APPEARANCE_DIR):
        if not fname.endswith(".npy"):
            continue
        path = os.path.join(APPEARANCE_DIR, fname)
        try:
            vector = np.load(path)
        except Exception:
            continue
        player_id = fname.split("_")[0]
        profiles.setdefault(player_id, []).append(vector)

    _appearance_cache = profiles
    _appearance_cache_time = current_time
    return profiles


def remove_player_data(player_id):
    global _known_faces_cache, _known_faces_cache_time, _appearance_cache, _appearance_cache_time
    prefix = f"{player_id}_"
    for fname in os.listdir(KNOWN_DIR):
        if fname.startswith(prefix):
            os.remove(os.path.join(KNOWN_DIR, fname))
    for fname in os.listdir(APPEARANCE_DIR):
        if fname.startswith(prefix):
            os.remove(os.path.join(APPEARANCE_DIR, fname))
    _known_faces_cache = None
    _known_faces_cache_time = 0
    _appearance_cache = None
    _appearance_cache_time = 0


def save_face_encoding(player_id, label, encoding):
    path = os.path.join(KNOWN_DIR, f"{player_id}_{label}.npy")
    np.save(path, encoding)
    global _known_faces_cache, _known_faces_cache_time
    _known_faces_cache = None
    _known_faces_cache_time = 0

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "activeProfiles": len(load_known_faces()),
        "appearanceProfiles": len(load_appearance_profiles())
    })

@app.route("/recognize", methods=["POST"])
def recognize():
    start_time = time.time()
    data = request.get_json()
    img_b64 = data.get("image")
    if not img_b64:
        return jsonify({"error": "Missing image"}), 400

    try:
        decode_start = time.time()
        img_data = base64.b64decode(img_b64)
        img_buffer = BytesIO(img_data)
        img = face_recognition.load_image_file(img_buffer)
        app.logger.info(f"Image decoded in {time.time() - decode_start:.2f}s")
    except Exception as e:
        app.logger.error(f"Failed to decode image: {str(e)}")
        return jsonify({"error": "Invalid image", "detail": str(e)}), 400

    appearance_vector = None
    try:
        appearance_vector = compute_appearance_vector(Image.fromarray(img))
    except Exception as e:
        app.logger.warning(f"Appearance vector computation failed: {e}")

    encoding_start = time.time()
    encodings = face_recognition.face_encodings(img)
    if encodings:
        app.logger.info(f"Face encoding extracted in {time.time() - encoding_start:.2f}s")
        unknown = encodings[0]

        load_start = time.time()
        known_faces = load_known_faces()
        app.logger.info(f"Known faces loaded in {time.time() - load_start:.2f}s (cached: {_known_faces_cache is not None})")

        compare_start = time.time()
        for pid, encs in known_faces.items():
            for known in encs:
                match = face_recognition.compare_faces([known], unknown, tolerance=0.45)
                if match[0]:
                    total_time = time.time() - start_time
                    app.logger.info(f"Player {pid} identified by face in {total_time:.2f}s")
                    return jsonify({"playerId": pid, "isNew": False, "matchType": "face"})
        app.logger.info(f"Face comparison finished in {time.time() - compare_start:.2f}s without match")
    else:
        app.logger.warning("No face found in image; attempting appearance match")

    pid, score = match_appearance_profile(appearance_vector)
    if pid:
        total_time = time.time() - start_time
        app.logger.info(f"Appearance match {pid} with score {score:.3f} in {total_time:.2f}s")
        return jsonify({"playerId": pid, "isNew": False, "matchType": "appearance", "confidence": score})

    if not encodings:
        return jsonify({"error": "No face found"}), 400

    total_time = time.time() - start_time
    app.logger.warning(f"No match found after {total_time:.2f}s")
    return jsonify({"error": "profile not found"}), 404


@app.route("/profiles", methods=["POST"])
def create_profile():
    data = request.get_json() or {}
    player_id = data.get("playerId")
    images = data.get("images", [])
    angles = data.get("angles", [])
    photos_payload = data.get("photos") or []

    app.logger.info(f"[DEBUG] Received profile request: playerId={player_id}, images_count={len(images) if images else 0}, angles_count={len(angles) if angles else 0}")

    if not player_id:
        app.logger.error(f"[ERROR] Missing playerId")
        return jsonify({"error": "playerId is required"}), 400
    
    if not photos_payload and (not images or len(images) == 0):
        app.logger.error(f"[ERROR] Missing photos or images array")
        return jsonify({"error": "photos or images array is required"}), 400

    remove_player_data(player_id)

    saved_faces = 0
    saved_appearance = 0
    errors = []

    if photos_payload:
        for idx, photo_entry in enumerate(photos_payload):
            try:
                faces, appearance = process_photo_entry(player_id, photo_entry, idx, angles)
                saved_faces += faces
                saved_appearance += appearance
            except Exception as e:
                app.logger.error(f"[ERROR] Failed to process photo {idx}: {e}", exc_info=True)
                errors.append(str(e))
        if saved_faces == 0 and saved_appearance == 0:
            return jsonify({"error": f"no valid photos processed: {errors}"}), 400
        app.logger.info(f"[SUCCESS] Profile created via payload for {player_id}: faces={saved_faces}, appearance={saved_appearance}")
        return jsonify({"success": True, "faces": saved_faces, "appearance": saved_appearance})

    # Legacy fallback for images + angles
    for idx, img_b64 in enumerate(images):
        try:
            if not img_b64:
                errors.append(f"Image {idx} is empty")
                app.logger.error(f"[ERROR] Image {idx} is empty")
                continue

            img_data = base64.b64decode(img_b64)
            img_buffer = BytesIO(img_data)
            img = face_recognition.load_image_file(img_buffer)
        except Exception as e:
            errors.append(f"Image {idx} decode error: {str(e)}")
            app.logger.error(f"[ERROR] Failed to decode image {idx}: {str(e)}", exc_info=True)
            continue

        encodings = face_recognition.face_encodings(img)
        if not encodings:
            errors.append(f"Image {idx} no face found")
            app.logger.error(f"[ERROR] No face found in image {idx}")
            continue

        label = angles[idx] if idx < len(angles) and angles[idx] else str(idx)
        save_face_encoding(player_id, label, encodings[0])
        saved_faces += 1
        app.logger.info(f"[SUCCESS] Saved encoding for player {player_id}, angle {label}")

    if saved_faces == 0:
        return jsonify({"error": f"no valid faces detected. Errors: {errors}"}), 400

    app.logger.info(f"[SUCCESS] Legacy profile created for player {player_id}, saved {saved_faces} encodings")
    return jsonify({"success": True, "faces": saved_faces})


def process_photo_entry(player_id, photo_entry, index, angles):
    angle = photo_entry.get("angle")
    if not angle:
        angle = angles[index] if index < len(angles) else str(index)
    original_b64 = photo_entry.get("original")
    if not original_b64:
        raise ValueError(f"Photo {index} missing original data")

    original_np, original_pil = decode_base64_image(original_b64)

    face_encoding = None
    face_b64 = photo_entry.get("face")
    if face_b64:
        face_np, _ = decode_base64_image(face_b64)
        face_encoding = extract_face_encoding(face_np)

    if face_encoding is None:
        face_encoding = extract_face_encoding(original_np)

    timestamp = int(photo_entry.get("capturedAt") or time.time() * 1000)
    label = f"{angle}_{timestamp}_{index}"

    faces_saved = 0
    if face_encoding is not None:
        save_face_encoding(player_id, label, face_encoding)
        faces_saved = 1
    else:
        app.logger.warning(f"[WARN] No face encoding extracted for photo {index}")

    appearance_saved = 0
    appearance_vector = None
    try:
        appearance_vector = compute_appearance_vector(original_pil)
    except Exception as e:
        app.logger.warning(f"[WARN] Failed to compute appearance vector for photo {index}: {e}")

    if appearance_vector is not None:
        save_appearance_vector(player_id, label, appearance_vector)
        appearance_saved = 1

    return faces_saved, appearance_saved


def decode_base64_image(img_b64):
    img_data = base64.b64decode(img_b64)
    pil = Image.open(BytesIO(img_data)).convert("RGB")
    return np.array(pil), pil


def extract_face_encoding(np_image):
    encodings = face_recognition.face_encodings(np_image)
    if not encodings:
        return None
    return encodings[0]


def save_appearance_vector(player_id, label, vector):
    path = os.path.join(APPEARANCE_DIR, f"{player_id}_{label}.npy")
    np.save(path, vector)
    global _appearance_cache, _appearance_cache_time
    _appearance_cache = None
    _appearance_cache_time = 0


def compute_appearance_vector(pil_image):
    resized = pil_image.resize((256, 256), Image.Resampling.BILINEAR)
    rgb_np = np.array(resized).astype("float32") / 255.0
    hsv_img = resized.convert("HSV")
    hsv_np = np.array(hsv_img).astype("float32") / 255.0

    mean_rgb = rgb_np.reshape(-1, 3).mean(axis=0)
    mean_hsv = hsv_np.reshape(-1, 3).mean(axis=0)
    hue_hist, _ = np.histogram(hsv_np[:, :, 0], bins=12, range=(0, 1), density=True)
    sat_hist, _ = np.histogram(hsv_np[:, :, 1], bins=5, range=(0, 1), density=True)

    vector = np.concatenate([mean_rgb, mean_hsv, hue_hist, sat_hist]).astype("float32")
    norm = np.linalg.norm(vector)
    if norm == 0:
        return None
    return vector / norm


def match_appearance_profile(vector):
    if vector is None:
        return None, None
    profiles = load_appearance_profiles()
    best_score = 0.0
    best_pid = None
    for pid, vectors in profiles.items():
        for known in vectors:
            score = cosine_similarity(vector, known)
            if score > best_score:
                best_score = score
                best_pid = pid
    if best_pid and best_score >= APPEARANCE_THRESHOLD:
        return best_pid, float(best_score)
    return None, None


def cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
