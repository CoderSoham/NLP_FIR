from flask import Flask, request, render_template, send_file, url_for, send_from_directory, abort, jsonify
import os
import uuid
from werkzeug.utils import secure_filename
from utils.audio_utils import process_audio_file, generate_fir_pdf
from config import UPLOAD_FOLDER, PROCESSED_FOLDER, MAX_CONTENT_LENGTH, ALLOWED_EXTENSIONS, DEBUG

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

@app.route('/', methods=['GET', 'POST'])
def index():
    data = {}
    if request.method == 'POST':
        if 'audio_file' not in request.files:
            return 'No file part'

        file = request.files['audio_file']
        if file.filename == '':
            return 'No selected file'

        if file:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs(PROCESSED_FOLDER, exist_ok=True)

            filename = secure_filename(file.filename)
            ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
            if ext not in ALLOWED_EXTENSIONS:
                abort(400, description='Unsupported file type')

            # Generate UUID-based filename to avoid collisions
            unique_name = f"{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
            file.save(filepath)

            # Process audio and generate plots & PDF
            data = process_audio_file(filepath, PROCESSED_FOLDER)

            # Pass filename to template for audio playback
            data['audio_filename'] = unique_name
            data['audio_url'] = url_for('serve_audio', filename=unique_name)

    return render_template('index.html', **data)

@app.route('/healthz', methods=['GET'])
def healthz():
    return jsonify({"status": "ok"})

@app.route('/download_fir')
def download_fir():
    pdf_path = os.path.join(PROCESSED_FOLDER, "fir_report.pdf")
    return send_file(pdf_path, as_attachment=True)

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/process', methods=['POST'])
def api_process():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported file type"}), 400
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)
    try:
        data = process_audio_file(filepath, PROCESSED_FOLDER)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=DEBUG)
