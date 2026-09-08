from flask import Flask, request, render_template, send_file, url_for, send_from_directory, abort, jsonify
import os
import re
import uuid
from werkzeug.utils import secure_filename
from utils.audio_utils import process_audio_file, generate_fir_pdf
from utils.retention import start_background_sweeper
from config import (UPLOAD_FOLDER, PROCESSED_FOLDER, MAX_CONTENT_LENGTH,
                    ALLOWED_EXTENSIONS, DEBUG, RETENTION_SECONDS,
                    RETENTION_SWEEP_SECONDS)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
start_background_sweeper([UPLOAD_FOLDER, PROCESSED_FOLDER],
                         max_age_seconds=RETENTION_SECONDS,
                         interval_seconds=RETENTION_SWEEP_SECONDS)

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

@app.route('/download_fir/<report_id>')
def download_fir(report_id):
    # Reports are per-request. This route previously served one shared
    # `fir_report.pdf`, so under any concurrency a caller could download
    # somebody else's report.
    if not re.fullmatch(r'[0-9a-f]{32}', report_id):
        abort(404)
    pdf_path = os.path.join(PROCESSED_FOLDER, f"fir_{report_id}.pdf")
    if not os.path.isfile(pdf_path):
        abort(404)
    return send_file(pdf_path, as_attachment=True,
                     download_name=f"FIR_{report_id[:8]}.pdf")

PLOT_KINDS = {'waveform', 'mfcc', 'pitch', 'entities'}

@app.route('/plot/<report_id>/<kind>')
def serve_plot(report_id, kind):
    """Serve one plot belonging to one request.

    Plots used to be written to fixed names under `static/plots/` and served by
    the static mount, so concurrent callers overwrote each other's audio and
    anyone could fetch the most recent call's waveform without uploading
    anything. They are now keyed by report_id and served from here, behind the
    same validation `download_fir` uses -- both path segments are
    user-controlled, so both are checked against a fixed set.
    """
    if kind not in PLOT_KINDS or not re.fullmatch(r'[0-9a-f]{32}', report_id):
        abort(404)
    path = os.path.join(PROCESSED_FOLDER, f"{kind}_{report_id}.png")
    if not os.path.isfile(path):
        abort(404)
    return send_file(path, mimetype='image/png')

@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serve an uploaded recording.

    Uploads live outside static/ now, so this route is the only way to reach
    them. `send_from_directory` rejects traversal, and names are uuid4 hex.
    """
    if not re.fullmatch(r'[0-9a-f]{32}\.[a-z0-9]{1,5}', filename):
        abort(404)
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
