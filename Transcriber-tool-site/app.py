import os
import re
from flask import Flask, request, render_template, send_file
import whisper
import torch
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["OUTPUT_FOLDER"] = "outputs"
# allow big lectures (e.g., up to 2 GB)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# Load Whisper once at startup (faster per request)
def load_whisper_model():
    # Force CPU by default (MPS can fail on sparse ops). Change to "mps" if it works for you.
    device = "cpu"
    try:
        if torch.backends.mps.is_available():
            # Try MPS; if it fails, we'll fall back to CPU
            return whisper.load_model("small", device="mps")
    except Exception as e:
        print("MPS failed, falling back to CPU:", e)
    return whisper.load_model("small", device=device)

model = load_whisper_model()

def transcribe_to_pdf(input_path: str) -> str:
    """
    Runs Whisper on the given file and writes a paragraph-formatted PDF.
    Returns the path to the created PDF.
    """
    # Transcribe (FFmpeg can handle mp3/wav/m4a/mp4/mov, etc.)
    result = model.transcribe(input_path, language="en", verbose=False)
    text = result["text"]

    # Build PDF path in /outputs with the same base name
    base = os.path.splitext(os.path.basename(input_path))[0]
    pdf_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{base}.pdf")

    # Your paragraph grouping logic (every 4 sentences)
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    sentences = re.split(r'(?<=[.!?]) +', text)
    paragraph = ""
    for i, sentence in enumerate(sentences, 1):
        paragraph += sentence + " "
        if i % 4 == 0:
            story.append(Paragraph(paragraph.strip(), styles["Normal"]))
            story.append(Spacer(1, 12))
            paragraph = ""
    if paragraph:
        story.append(Paragraph(paragraph.strip(), styles["Normal"]))

    doc.build(story)
    return pdf_path

def allowed_file(filename: str) -> bool:
    # accept common audio/video; FFmpeg will decode
    allowed = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4", ".mov", ".mkv", ".aac"}
    ext = os.path.splitext(filename.lower())[1]
    return ext in allowed

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if not file or file.filename == "":
            return "No selected file", 400
        if not allowed_file(file.filename):
            return "Unsupported file type", 400

        # Save upload
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(upload_path)

        # Transcribe → PDF
        pdf_path = transcribe_to_pdf(upload_path)

        # Return the PDF as a download
        return send_file(pdf_path, as_attachment=True)

    return render_template("index.html")

if __name__ == "__main__":
    # Debug server for local use
    app.run(debug=True)
