import os
import sqlite3
from datetime import datetime
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for, session,
    flash, send_from_directory
)
from werkzeug.security import generate_password_hash, check_password_hash

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


# -------------------------------------------------
# Flask app setup
# -------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = "supersecretkey-change-this"
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Database + output paths
INSTANCE_DIR = os.path.join(APP_ROOT, "instance")
os.makedirs(INSTANCE_DIR, exist_ok=True)
DATABASE = os.path.join(INSTANCE_DIR, "users.db")

OUTPUTS_DIR = os.path.join(APP_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app.config["DATABASE"] = DATABASE
app.config["OUTPUTS_DIR"] = OUTPUTS_DIR


# -------------------------------------------------
# FIX: Make datetime available to ALL templates
# -------------------------------------------------
@app.context_processor
def inject_now():
    return {"datetime": datetime}


# -------------------------------------------------
# Database helper
# -------------------------------------------------
def get_db_connection():
    conn = sqlite3.connect(app.config["DATABASE"])
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()


# -------------------------------------------------
# Login required decorator
# -------------------------------------------------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            flash("Please login first.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


# -------------------------------------------------
# Load Stable Diffusion Model once
# -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

MODEL_ID = "runwayml/stable-diffusion-v1-5"

print("[INFO] Loading diffusion model...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

if device == "cuda":
    pipe.enable_model_cpu_offload()
else:
    pipe.to("cpu")

print("[INFO] Model loaded successfully.")


# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")


# ---------- Register ----------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":

        username = request.form["username"].strip()
        password = request.form["password"]

        if not username or not password:
            flash("All fields required.", "danger")
            return render_template("register.html")

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            hashed = generate_password_hash(password)
            cur.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (username, hashed, datetime.utcnow().isoformat())
            )
            conn.commit()
            conn.close()
            flash("Registration successful. Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists.", "danger")
            conn.close()
            return render_template("register.html")

    return render_template("register.html")


# ---------- Login ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":

        username = request.form["username"].strip()
        password = request.form["password"]

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session["user"] = username
            flash("Logged in successfully.", "success")
            return redirect(url_for("predict"))

        flash("Invalid username or password.", "danger")
        return render_template("login.html")

    return render_template("login.html")


# ---------- Logout ----------
@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out.", "info")
    return redirect(url_for("home"))


# ---------- Generate Sketch ----------
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        username = session["user"]

        gender = request.form.get("gender", "")
        age_group = request.form.get("age_group", "")
        hair = request.form.get("hair", "")
        face_shape = request.form.get("face_shape", "")
        eyes = request.form.get("eyes", "")
        nose = request.form.get("nose", "")
        expression = request.form.get("expression", "")

        beard = ""
        mustache = ""
        earrings = ""

        if gender.lower() == "male":
            if request.form.get("has_beard") == "yes":
                beard = request.form.get("beard_desc", "")
            if request.form.get("has_mustache") == "yes":
                mustache = request.form.get("mustache_desc", "")

        if gender.lower() == "female":
            if request.form.get("has_earrings") == "yes":
                earrings = request.form.get("earrings_desc", "")

        extras = request.form.get("extras", "").lower()
        if extras in ["no", "none", "nil", "nothing", ""]:
            extras = ""

        features = [
            hair,
            f"{face_shape} face",
            eyes,
            f"{nose} nose",
            f"{expression} expression",
            beard,
            mustache,
            earrings,
            extras
        ]

        # remove empty
        features = [f for f in features if f.strip() != ""]

        all_features = ", ".join(features)

        prompt = (
            f"pencil sketch portrait of a {age_group.lower()} {gender.lower()} "
            f"with {all_features}. realistic pencil drawing, "
            f"detailed shading, monochrome on white background."
        )

        # generate sketch
        try:
            result = pipe(prompt, num_inference_steps=25, guidance_scale=8.5)
            img = result.images[0]
        except Exception as e:
            flash(f"Error generating image: {e}", "danger")
            return redirect(url_for("predict"))

        # save to user-specific folder
        user_folder = os.path.join(OUTPUTS_DIR, username)
        os.makedirs(user_folder, exist_ok=True)

        filename = f"sketch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = os.path.join(user_folder, filename)
        img.save(save_path)

        return redirect(url_for("show_result", username=username, filename=filename))

    return render_template("predict.html")


# ---------- Show generated sketch ----------
@app.route("/result/<username>/<filename>")
@login_required
def show_result(username, filename):
    if session["user"] != username:
        flash("Unauthorized.", "danger")
        return redirect(url_for("home"))

    file_url = url_for("serve_output", username=username, filename=filename)
    return render_template("result.html", filename=filename, file_url=file_url)


# ---------- Serve images ----------
@app.route("/outputs/<username>/<filename>")
@login_required
def serve_output(username, filename):
    if session["user"] != username:
        flash("Unauthorized.", "danger")
        return redirect(url_for("home"))

    user_folder = os.path.join(OUTPUTS_DIR, username)
    return send_from_directory(user_folder, filename)


# ---------- Gallery page ----------
@app.route("/gallery")
@login_required
def gallery():
    username = session["user"]
    user_folder = os.path.join(OUTPUTS_DIR, username)

    if not os.path.exists(user_folder):
        images = []
    else:
        images = sorted(os.listdir(user_folder), reverse=True)

    return render_template("profile.html", images=images, user=username)


# -------------------------------------------------
# Run App
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
