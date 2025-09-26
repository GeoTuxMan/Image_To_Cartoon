# app.py
import io
from flask import Flask, request, render_template_string, send_file, jsonify
import cv2
import numpy as np
from cartoon import image_to_cartoon

app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<title>Cartoonifier</title>
<h1>Upload image to cartoonify</h1>
<form method=post enctype=multipart/form-data action="/cartoon">
  <input type=file name=file accept="image/*">
  <input type=submit value="Start">
</form>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/cartoon", methods=["POST"])
def cartoon_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    file = request.files["file"]
    # read into numpy array
    file_bytes = file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "cannot decode image"}), 400

    cartoon = image_to_cartoon(img)

    # encode to JPEG in memory
    ret, buf = cv2.imencode(".jpg", cartoon)
    if not ret:
        return jsonify({"error": "encode failed"}), 500
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg", as_attachment=False, download_name="cartoon.jpg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
