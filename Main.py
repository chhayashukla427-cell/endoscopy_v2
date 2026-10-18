import Investigate
import base64
import os
import shutil

import cv2
import numpy as np
from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = 'super secret key'.encode('utf8')
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_FOLDER = os.path.join("static", "user_uploads", "original")
ALLOWED_EXTENSIONS = {'tiff', 'tif', 'png', 'jpg', 'jpeg'}
SAMPLE_IMAGES = tuple(f"{index}.png" for index in range(1, 13))

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# check if the file is in accepted format
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

"""
Main page -> choose files to upload
"""
@app.route('/')
def main_page():
    return render_template("index.html")

"""
Investigate uploaded image
"""
@app.route('/upload', methods = ["GET", "POST"])
def upload_page():
    if request.method != "POST":
        return redirect(url_for("main_page"))

    image_file = request.files.get("file_input")
    if image_file is None or not image_file.filename:
        flash("Choose an image before continuing.")
        return redirect(url_for("main_page"))

    filename = secure_filename(image_file.filename)
    if not filename or not allowed_file(filename):
        flash("Choose a PNG, JPG or TIFF image.")
        return redirect(url_for("main_page"))

    image_data = image_file.read()
    image = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        flash("The selected file could not be read as an image.")
        return redirect(url_for("main_page"))

    clear_past_uploads()
    upload_dir = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    with open(upload_dir, "wb") as saved_file:
        saved_file.write(image_data)
    image_backend = os.path.relpath(upload_dir, app.root_path).replace(os.sep, "/")
    return render_template("investigate.html", image_backend=image_backend)


@app.route("/upload-sample", methods=["POST"])
def upload_sample():
    filename = request.form.get("sample", "")
    if filename not in SAMPLE_IMAGES:
        flash("That sample image is not available.")
        return redirect(url_for("show_gallery"))

    sample_path = os.path.join(app.root_path, "static", "sample_images", filename)
    if not os.path.isfile(sample_path):
        app.logger.error("Configured sample image is missing: %s", filename)
        flash("That sample image is currently unavailable.")
        return redirect(url_for("show_gallery"))

    clear_past_uploads()
    upload_dir = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    shutil.copyfile(sample_path, upload_dir)
    image_backend = os.path.relpath(upload_dir, app.root_path).replace(os.sep, "/")
    return render_template("investigate.html", image_backend=image_backend)

"""
Classify chosen patch
"""
@app.route("/classify", methods = ["POST" , "GET"])
def classify_patch():
    if request.method != "POST":
        return redirect(url_for("main_page"))

    try:
        x = int(request.form["form_x"])
        y = int(request.form["form_y"])
    except (KeyError, ValueError):
        flash("Select a point on the image to classify a patch.")
        return redirect(url_for("main_page"))

    uploaded_files = os.listdir(app.config["UPLOAD_FOLDER"])
    if not uploaded_files:
        flash("Upload an image before selecting a patch.")
        return redirect(url_for("main_page"))

    upload_dir = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_files[0])
    img, prediction = Investigate.classify_patch(src=upload_dir, x=x, y=y)

    success, encoded_image = cv2.imencode(".png", img)
    if not success:
        raise OSError("The classification result image could not be encoded.")
    image_backend = "data:image/png;base64," + base64.b64encode(encoded_image).decode("ascii")
    return render_template(
        "investigate.html",
        image_backend=image_backend,
        prediction=prediction,
    )

"""
Clear all past uploaded images
"""
def clear_past_uploads():
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    for files in os.listdir(app.config["UPLOAD_FOLDER"]):
        path = os.path.join(app.config["UPLOAD_FOLDER"], files)
        if os.path.isfile(path):
            os.remove(path)

"""
Show Gallery
"""
@app.route("/gallery")
def show_gallery():
    return render_template("gallery.html", samples=SAMPLE_IMAGES)


@app.errorhandler(413)
def request_too_large(error):
    flash("That image is too large. Choose an image smaller than 16 MB.")
    return redirect(url_for("main_page"))

@app.route("/About")
def about():
    return "About us"


if(__name__ == "__main__"):
    port = int(os.environ.get("PORT", 5001))
    app.run(port=port)  
