import os
import uuid
import torch
import librosa
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive 'Agg'
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from sentence_transformers import SentenceTransformer, util
import spacy
from fpdf import FPDF
import librosa.display
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
from config import (PROCESSED_FOLDER, WHISPER_MODEL_NAME, FORCE_CPU,
                    MAX_AUDIO_SECONDS, EMOTION_MIN_CONFIDENCE)
from utils.plots import plot_path, generate_entity_plot
from utils.dispatch import get_dispatch_suggestion
from utils.signals import augment_actions
from utils.severity import severity_score, severity_label
from utils.summary import length_budget, is_informative
from utils.llm import extract_incident
from utils import gpu
from utils.asr import transcribe as asr_transcribe
from utils.audio_clean import clean_audio
from typing import Optional, Tuple

_DEVICE = None

def resolve_device():
    """Resolve the torch device once, honouring FORCE_CPU.

    `FORCE_CPU` was defined in config.py and read by nothing -- every loader
    hardcoded 'cpu' / -1 at its call site, so a GPU on the host went unused and
    the setting was a knob wired to nothing.

    Defaults to CPU. Both BART-large models together do not fit on a 6 GB card,
    so enabling CUDA is a claim about your hardware that only you can make.
    """
    global _DEVICE
    if _DEVICE is None:
        if not FORCE_CPU and torch.cuda.is_available():
            _DEVICE = 'cuda'
        else:
            _DEVICE = 'cpu'
    return _DEVICE

def pipeline_device():
    """transformers.pipeline takes an int: -1 for CPU, else the CUDA ordinal."""
    return 0 if resolve_device() == 'cuda' else -1

def unload_nlp_models():
    """Release the classical transformer pipelines from the GPU.

    These are the zero-shot classifier, the two summarisers, the sentiment and
    emotion heads and the sentence embedder -- together roughly 4 GB of
    weights. With FORCE_CPU=0 they all land on the GPU, and nothing evicted
    them, so the LLM stage could never allocate and reported itself unavailable
    on every request while working perfectly in isolation.

    They are registered as one slot because they run as one phase, between
    transcription and extraction.
    """
    global embedder, nlp, t5_tokenizer, t5_model, EMERGENCY_CLASSIFIER
    global SEVERITY_CLASSIFIER, NER_MODEL, SENTENCE_MODEL, SUMMARIZER
    global SENTIMENT_ANALYZER, EMOTION_DETECTOR

    if resolve_device() != 'cuda':
        return False
    held = any(m is not None for m in (
        embedder, t5_model, EMERGENCY_CLASSIFIER, SUMMARIZER,
        SENTIMENT_ANALYZER, EMOTION_DETECTOR, SENTENCE_MODEL))
    if not held:
        return False

    embedder = None
    t5_tokenizer = t5_model = None
    EMERGENCY_CLASSIFIER = SEVERITY_CLASSIFIER = None
    SENTENCE_MODEL = SUMMARIZER = None
    SENTIMENT_ANALYZER = EMOTION_DETECTOR = None
    # spaCy stays: it is CPU-only and cheap to keep.
    gpu.empty_cache()
    return True


gpu.register('nlp', unload_nlp_models)

# Initialize models as None for lazy loading
embedder = None
nlp = None
t5_tokenizer = None
t5_model = None
EMERGENCY_CLASSIFIER = None
SEVERITY_CLASSIFIER = None
NER_MODEL = None
SENTENCE_MODEL = None
SUMMARIZER = None
SENTIMENT_ANALYZER = None
EMOTION_DETECTOR = None

def load_embedder():
    global embedder
    gpu.acquire('nlp')
    if embedder is None:
        embedder = SentenceTransformer('all-MiniLM-L6-v2', device=resolve_device())
    return embedder

def load_nlp():
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_sm", disable=['parser', 'textcat'])
    return nlp

def load_t5():
    global t5_tokenizer, t5_model
    gpu.acquire('nlp')
    if t5_tokenizer is None or t5_model is None:
        t5_model_name = "t5-small"
        t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        t5_model.eval()  # Set to evaluation mode
    return t5_tokenizer, t5_model

def load_emergency_classifier():
    global EMERGENCY_CLASSIFIER
    gpu.acquire('nlp')
    if EMERGENCY_CLASSIFIER is None:
        EMERGENCY_CLASSIFIER = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=pipeline_device(),
            framework="pt"  # Use PyTorch backend
        )
    return EMERGENCY_CLASSIFIER

def load_severity_classifier():
    """Zero-shot severity scoring.

    Reuses the bart-large-mnli pipeline rather than loading a second model.
    This was previously `microsoft/deberta-v3-base`, which has no NLI head --
    the zero-shot pipeline attaches a randomly initialised classification head
    to it, so every severity score it produced was untrained noise. Reusing
    the MNLI model fixes the correctness problem and saves ~400 MB of RAM.
    """
    global SEVERITY_CLASSIFIER
    if SEVERITY_CLASSIFIER is None:
        SEVERITY_CLASSIFIER = load_emergency_classifier()
    return SEVERITY_CLASSIFIER

def load_ner_model():
    global NER_MODEL
    if NER_MODEL is None:
        NER_MODEL = spacy.load("en_core_web_sm", disable=['parser', 'textcat'])
    return NER_MODEL

def load_sentence_model():
    global SENTENCE_MODEL
    gpu.acquire('nlp')
    if SENTENCE_MODEL is None:
        SENTENCE_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device=resolve_device())
    return SENTENCE_MODEL

def load_summarizer():
    global SUMMARIZER
    gpu.acquire('nlp')
    if SUMMARIZER is None:
        SUMMARIZER = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=pipeline_device(),
            framework="pt"  # Use PyTorch backend
        )
    return SUMMARIZER

def load_sentiment_analyzer():
    global SENTIMENT_ANALYZER
    gpu.acquire('nlp')
    if SENTIMENT_ANALYZER is None:
        SENTIMENT_ANALYZER = pipeline(
            "sentiment-analysis",
            model="finiteautomata/bertweet-base-sentiment-analysis",
            device=pipeline_device(),
            framework="pt"  # Use PyTorch backend
        )
    return SENTIMENT_ANALYZER

def load_emotion_detector():
    global EMOTION_DETECTOR
    gpu.acquire('nlp')
    if EMOTION_DETECTOR is None:
        EMOTION_DETECTOR = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=pipeline_device(),
            framework="pt"  # Use PyTorch backend
        )
    return EMOTION_DETECTOR

# Sanitize Unicode for fpdf
def sanitize_text(text):
    return text.encode('latin-1', errors='ignore').decode('latin-1')

def clean_summary(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Capitalize first letter of each sentence
    text = '. '.join(s.strip().capitalize() for s in text.split('.'))
    return text

# Summarization function
def summarize_text(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "No content to summarize."

    # Ensure T5 tokenizer/model are loaded
    global t5_tokenizer, t5_model
    if t5_tokenizer is None or t5_model is None:
        t5_tokenizer, t5_model = load_t5()

    # Create a more specific prompt for emergency situations
    prompt = "summarize this emergency call in a clear and concise way, focusing on the type of emergency, location, and key details: "
    input_text = prompt + text
    
    # Encode with longer max length to capture more context
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
    
    # Generate summary with adjusted parameters
    with torch.inference_mode():
        t5_max, t5_min = length_budget(len(text.split()), ceiling=200)
        summary_ids = t5_model.generate(
            input_ids,
            max_length=t5_max,
            min_length=t5_min,
            length_penalty=1.5,  # Balanced length penalty
            num_beams=5,     # More beams for better quality
            early_stopping=True,
            no_repeat_ngram_size=3  # Prevent repetition
        )
    
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Post-process the summary
    summary = clean_summary(summary)
    
    # Add emergency context if not present
    if not any(word in summary.lower() for word in ['emergency', 'accident', 'fire', 'medical', 'police', 'ambulance']):
        summary = "Emergency Call Summary: " + summary
    
    return summary

def summarize_text_chunked(text: str) -> str:
    """Higher quality summarization using BART with chunking and T5 fallback."""
    text = (text or "").strip()
    if not text:
        return ""
    summarizer = load_summarizer()
    # Chunk into ~700-char windows with 120-char overlap for context
    max_len = 700
    overlap = 120
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+max_len])
        i += max_len - overlap
    summaries = []
    try:
        with torch.inference_mode():
            for ch in chunks:
                ch = ch.strip()
                if not ch:
                    continue
                ch_max, ch_min = length_budget(len(ch.split()))
                out = summarizer(ch, max_length=ch_max, min_length=ch_min,
                                 do_sample=False)[0]['summary_text']
                summaries.append(out)
    except Exception:
        # If BART errors, fall back to existing T5 summarizer
        return summarize_text(text)
    # Combine summaries
    combined = " ".join(summaries)
    # Final pass to tighten
    try:
        with torch.inference_mode():
            fin_max, fin_min = length_budget(len(combined.split()))
            final = summarizer(combined, max_length=fin_max, min_length=fin_min,
                               do_sample=False)[0]['summary_text']
            return clean_summary(final)
    except Exception:
        return clean_summary(combined)

def is_summary_informative(summary: str, source: str) -> bool:
    """Whether a summary is worth showing. See utils.summary for the rule."""
    return is_informative(summary, source)

def augment_actions_from_transcript(transcript: str, base_response: dict) -> dict:
    """Augment recommended actions from high-signal phrases in the transcript.

    Delegates to utils.signals, which matches whole words. This used to do
    `substring in transcript` with 'ex' and 'car' among the keywords -- see the
    module docstring there for what that did to real calls.
    """
    return augment_actions(transcript, base_response)

# Command templates
known_commands = [
    "send ambulance", "send firetruck", "send police",
    "there is an accident", "house on fire", "person not breathing",
    "there is a fire", "medical emergency", "need help immediately",
    "person unconscious", "vehicle crash", "building collapse",
    "gas leak", "gunshot heard", "hostage situation",
    "earthquake response", "flood rescue", "emergency medical team",
    "fire in the kitchen", "traffic accident on highway"
]

def get_known_embeddings():
    embedder = load_embedder()
    return embedder.encode(known_commands, convert_to_tensor=True)

# Emergency response templates with ML-enhanced features
EMERGENCY_RESPONSES = {
    'medical': {
        'priority': 'high',
        'suggestions': [
            'Dispatch medical team immediately',
            'Prepare emergency medical equipment',
            'Alert nearest hospital',
            'Coordinate with medical professionals'
        ],
        'required_resources': ['ambulance', 'medical_team', 'first_aid'],
        'response_time': 'immediate'
    },
    'fire': {
        'priority': 'high',
        'suggestions': [
            'Dispatch fire department immediately',
            'Evacuate affected area',
            'Coordinate with fire safety team',
            'Prepare fire suppression equipment'
        ],
        'required_resources': ['fire_truck', 'fire_team', 'evacuation_equipment'],
        'response_time': 'immediate'
    },
    'police': {
        'priority': 'high',
        'suggestions': [
            'Dispatch police units',
            'Secure the area',
            'Coordinate with law enforcement',
            'Document the situation'
        ],
        'required_resources': ['police_units', 'investigation_team'],
        'response_time': 'immediate'
    },
    'accident': {
        'priority': 'medium',
        'suggestions': [
            'Assess accident severity',
            'Coordinate with relevant authorities',
            'Secure the accident site',
            'Provide immediate assistance'
        ],
        'required_resources': ['emergency_team', 'traffic_control'],
        'response_time': 'asap'
    }
}

def get_emergency_type(text):
    """Optimized emergency classification using transformer models"""
    if not text or len(text.strip()) < 3:
        return "unknown"  # Return unknown for empty or very short text
    
    # Truncate text to prevent token length issues
    text = text[:500]  # Limit to 500 characters
        
    try:
        # Get base classification
        classifier = load_emergency_classifier()
        candidate_labels = ["medical emergency", "fire emergency", "police emergency", "accident"]
        base_result = classifier(text, candidate_labels=candidate_labels)
        
        # Get sentiment and emotion for context
        sentiment_analyzer = load_sentiment_analyzer()
        emotion_detector = load_emotion_detector()
        
        try:
            sentiment = sentiment_analyzer(text[:128])[0]  # Limit to 128 tokens for sentiment
        except Exception:
            sentiment = {"label": "NEU", "score": 0.5}
            
        try:
            emotion = emotion_detector(text[:128])[0]  # Limit to 128 tokens for emotion
        except Exception:
            emotion = {"label": "neutral", "score": 0.5}
        
        # Extract named entities for additional context
        ner_model = load_ner_model()
        doc = ner_model(text)
        entities = [ent.text for ent in doc.ents]
        
        # Combine all signals for final classification
        emergency_scores = {
            'medical': 0,
            'fire': 0,
            'police': 0,
            'accident': 0
        }
        
        # Update scores based on classification
        emergency_type = base_result['labels'][0].split()[0]
        emergency_scores[emergency_type] += base_result['scores'][0]
        
        # Adjust scores based on sentiment and emotion
        if sentiment['label'] == 'NEG' and sentiment['score'] > 0.7:
            emergency_scores[emergency_type] += 0.2
        
        if emotion['label'] in ['fear', 'anxiety']:
            emergency_scores[emergency_type] += 0.15
        
        # Adjust based on entities
        for entity in entities:
            if any(medical_term in entity.lower() for medical_term in ['hospital', 'doctor', 'ambulance']):
                emergency_scores['medical'] += 0.1
            elif any(fire_term in entity.lower() for fire_term in ['fire', 'smoke', 'burning']):
                emergency_scores['fire'] += 0.1
            elif any(police_term in entity.lower() for police_term in ['police', 'officer', 'crime']):
                emergency_scores['police'] += 0.1
            elif any(accident_term in entity.lower() for accident_term in ['accident', 'crash', 'collision']):
                emergency_scores['accident'] += 0.1
        
        # Get final classification
        final_type = max(emergency_scores.items(), key=lambda x: x[1])[0]
        return final_type
        
    except Exception as e:
        print(f"Error in emergency classification: {str(e)}")
        return "unknown"  # Return unknown if any part of the classification fails

def assess_severity(text, entities):
    """Assess severity from the transcript and the entities already extracted.

    The arithmetic lives in utils.severity as a pure function so it can be
    tested without loading a model. This function is the part that needs one.
    """
    if not text or len(text.strip()) < 3:
        return "low"

    text = text[:500]

    try:
        classifier = load_severity_classifier()
        result = classifier(text, candidate_labels=["high", "medium", "low"])
        distribution = dict(zip(result["labels"], result["scores"]))

        try:
            sentiment = load_sentiment_analyzer()(text[:128])[0]
        except Exception:
            sentiment = {"label": "NEU", "score": 0.5}
        try:
            emotion = load_emotion_detector()(text[:128])[0]
        except Exception:
            emotion = {"label": "neutral", "score": 0.5}

        score, _parts = severity_score(
            distribution, sentiment, emotion, entities,
            emotion_min_confidence=EMOTION_MIN_CONFIDENCE)
        return severity_label(score)

    except Exception as e:
        print(f"Error in severity assessment: {str(e)}")
        return "low"

def get_emergency_response(emergency_type):
    """Get ML-enhanced emergency response"""
    response = EMERGENCY_RESPONSES.get(emergency_type, {
        'priority': 'medium',
        'suggestions': ['Assess the situation', 'Coordinate with relevant authorities'],
        'required_resources': ['emergency_team'],
        'response_time': 'asap'
    })
    
    # Add ML-based response time estimation
    response['estimated_response_time'] = calculate_response_time(emergency_type, response['priority'])
    
    return response

def calculate_response_time(emergency_type, priority):
    """Calculate estimated response time using ML"""
    base_times = {
        'high': 5,  # minutes
        'medium': 15,
        'low': 30
    }
    
    # Adjust based on emergency type
    type_multipliers = {
        'medical': 0.8,  # Faster response for medical
        'fire': 0.9,
        'police': 1.0,
        'accident': 1.2
    }
    
    return base_times[priority] * type_multipliers.get(emergency_type, 1.0)

def process_audio_file(input_path, output_folder):
    """Process audio file and generate analysis"""
    try:
        # Load audio file with specific parameters for Whisper
        audio, sr = librosa.load(input_path, sr=16000, mono=True)  # Whisper expects 16kHz mono audio

        # Trim excessively long audio to bound processing time. The cap is
        # deliberate -- Whisper is roughly linear in duration on CPU, so one
        # long recording would otherwise occupy the only worker for minutes.
        #
        # What was missing is saying so. Two of the three sample calls are over
        # the cap, one losing 80% of the conversation, and the report presented
        # the result as an analysis of the whole call.
        max_seconds = MAX_AUDIO_SECONDS
        source_duration = len(audio) / float(sr)
        truncated = source_duration > max_seconds
        if truncated:
            audio = audio[: int(sr * max_seconds)]
        analysed_duration = len(audio) / float(sr)

        # Identifies every artefact this request produces, and is generated
        # before the first of them is written. It used to be created after
        # visualisation, which is why the plots kept fixed shared filenames
        # long after the PDF stopped having one.
        report_id = uuid.uuid4().hex

        # Condition the audio before recognition. Conservative on purpose --
        # see utils.audio_clean for why aggressive denoising hurts ASR.
        audio, audio_report = clean_audio(audio, sr)

        # Generate visualizations from what was actually transcribed, not from
        # the raw file, so the plots describe the analysed signal.
        plots = generate_visualizations(audio, sr, output_folder, report_id)

        # faster-whisper takes the array directly, so the intermediate WAV that
        # every request used to write and delete is gone.
        asr_result = asr_transcribe(audio)
        transcription = asr_result["text"]
        lang = asr_result["language"]

        # If non-English, translate to English for the downstream stages.
        translated_text = transcription
        if lang and lang != 'en':
            try:
                translated_text = asr_transcribe(audio, translate=True)["text"]
            except Exception:
                translated_text = transcription
        
        # Named entities first -- severity weights them, so they have to exist
        # before it runs. Previously severity was called with a hardcoded []
        # and the entity term of its score was dead on every request.
        ner_model = load_ner_model()
        doc = ner_model(translated_text)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        if generate_entity_plot(entities, output_folder, report_id):
            plots.append('entities')

        # Get emergency type and severity
        emergency_type = get_emergency_type(translated_text)
        severity = assess_severity(translated_text, entities)
        
        # Generate response
        response = get_emergency_response(emergency_type)
        response = augment_actions_from_transcript(translated_text, response)
        response_time = calculate_response_time(emergency_type, response['priority'])
        
        # Generate summary
        # Prefer high-quality chunked summarization
        summary = summarize_text_chunked(translated_text)
        if not is_summary_informative(summary, translated_text):
            # Fallback to T5; if still low-value, leave empty for UI to hide
            summary = summarize_text(translated_text)
            if not is_summary_informative(summary, translated_text):
                summary = ""

        # Simple location extraction heuristic (reuses the doc parsed above)
        probable_location = None
        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC", "FAC"):
                probable_location = ent.text
                break

        # Best match against known commands using embeddings
        embedder = load_embedder()
        with torch.no_grad():
            known_emb = get_known_embeddings()
            query_emb = embedder.encode([translated_text], convert_to_tensor=True)
            cos_scores = util.cos_sim(query_emb, known_emb)[0]
            top_idx = int(torch.argmax(cos_scores).item())
            best_match = known_commands[top_idx]
            score = float(cos_scores[top_idx].item())

        # Sentiment and emotion (best effort)
        try:
            sentiment = load_sentiment_analyzer()(translated_text[:256])[0]
        except Exception:
            sentiment = {"label": "NEU", "score": 0.5}
        try:
            emotion = load_emotion_detector()(translated_text[:256])[0]
        except Exception:
            emotion = {"label": "neutral", "score": 0.5}
        
        # Generate PDF report
        # Dispatch suggestion
        dispatch = get_dispatch_suggestion(emergency_type, probable_location)

        data = {
            'report_id': report_id,
            'transcription': transcription,
            'translated_text': translated_text if translated_text != transcription else None,
            'language': lang,
            'emergency_type': emergency_type,
            'severity': severity,
            'response': response,
            'response_time': response_time,
            'summary': summary,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'entities': entities,
            'plots': plots,
            'asr': {k: v for k, v in asr_result.items() if k != 'segments'},
            'asr_segments': asr_result['segments'],
            'audio_report': audio_report,
            'source_duration_s': round(source_duration, 1),
            'analysed_duration_s': round(analysed_duration, 1),
            'truncated': truncated,
            'best_match': best_match,
            'score': score,
            'sentiment': sentiment,
            'emotion': emotion,
            'emergency_response': response,
            'probable_location': probable_location,
            'dispatch': dispatch
        }
        
        # LLM enrichment. Runs after the classical path, never instead of it:
        # the deterministic stages are the floor, and both results are kept so
        # a disagreement between them can be surfaced rather than hidden.
        llm_record, llm_meta = extract_incident(
            translated_text, truncated=truncated,
            source_seconds=source_duration, analysed_seconds=analysed_duration)
        data['llm'] = llm_record
        data['llm_meta'] = llm_meta
        data['disagreements'] = compare_classifications(data, llm_record)

        data['fir_pdf'] = generate_fir_pdf(data)

        return data
        
    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        raise  # Re-raise the exception to handle it in the Flask route

def compare_classifications(classical, llm_record):
    """Where the classical pipeline and the LLM disagree.

    A disagreement is information, not an error. Two independent methods
    reaching different conclusions about an emergency call is exactly the case
    a human should look at, so it is reported rather than resolved by picking a
    winner.
    """
    if not llm_record:
        return []

    found = []
    llm_type = llm_record.get('incident_type')
    if llm_type and llm_type not in ('unknown', 'other') and llm_type != classical.get('emergency_type'):
        found.append({
            'field': 'emergency_type',
            'classical': classical.get('emergency_type'),
            'llm': llm_type,
        })

    # The LLM has a 'critical' band the classical scorer does not; fold it in
    # before comparing so the two are on the same scale.
    llm_sev = llm_record.get('severity')
    normalised = {'critical': 'high'}.get(llm_sev, llm_sev)
    if normalised and normalised != 'unknown' and normalised != classical.get('severity'):
        found.append({
            'field': 'severity',
            'classical': classical.get('severity'),
            'llm': llm_sev,
        })

    # The keyword matcher flags weapons on any mention; the LLM is asked to
    # ignore figures of speech. This is the disagreement that matters most.
    keyword_weapons = bool((classical.get('response') or {}).get('signals', {}).get('weapons'))
    llm_weapons = bool(llm_record.get('weapons'))  # now [{item, quote}, ...]
    if keyword_weapons != llm_weapons:
        found.append({
            'field': 'weapons',
            'classical': 'mentioned' if keyword_weapons else 'not mentioned',
            'llm': ([w.get('item') for w in llm_record.get('weapons') or []]
                    or 'none present'),
        })

    classical_location = classical.get('probable_location')
    llm_location = llm_record.get('location')
    if bool(classical_location) != bool(llm_location):
        found.append({
            'field': 'location',
            'classical': classical_location,
            'llm': llm_location,
        })

    return found

def generate_visualizations(audio, sr, output_folder, report_id):
    """Generate audio visualizations. Returns the plot kinds written."""
    os.makedirs(output_folder, exist_ok=True)

    # Waveform plot
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title('Waveform')
    plt.tight_layout()
    plt.savefig(plot_path(output_folder, 'waveform', report_id))
    plt.close()

    # MFCC plot
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC')
    plt.tight_layout()
    plt.savefig(plot_path(output_folder, 'mfcc', report_id))
    plt.close()

    # Pitch plot
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    plt.figure(figsize=(12, 4))
    plt.plot(pitches)
    plt.title('Pitch')
    plt.tight_layout()
    plt.savefig(plot_path(output_folder, 'pitch', report_id))
    plt.close()

    return ['waveform', 'mfcc', 'pitch']

def generate_fir_pdf(data):
    """Generate advanced PDF report"""
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.multi_cell(0, 10, sanitize_text("🚨 Emergency Response Report\n\n"))
    pdf.set_font("Arial", size=12)
    
    # Timestamp
    pdf.multi_cell(0, 10, sanitize_text(f"Generated at: {data['timestamp']}\n\n"))
    
    # Emergency Analysis
    pdf.set_font("Arial", 'B', 14)
    pdf.multi_cell(0, 10, sanitize_text("Emergency Analysis\n"))
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, sanitize_text(f"Type: {data['emergency_type'].title()}"))
    pdf.multi_cell(0, 10, sanitize_text(f"Severity: {data['severity'].title()}"))
    pdf.multi_cell(0, 10, sanitize_text(f"Priority: {data['response']['priority'].title()}"))
    # Labelled as a protocol target, not an ETA. It comes from the priority
    # alone and has nothing to do with the station or the caller's location; the
    # report used to print it beside the dispatch ETA as if they were comparable.
    pdf.multi_cell(0, 10, sanitize_text(
        f"Protocol target for this priority: {data['response']['estimated_response_time']} minutes"))

    dispatch = data.get('dispatch') or {}
    if dispatch.get('basis') == 'location_match':
        eta = dispatch.get('eta_min')
        pdf.multi_cell(0, 10, sanitize_text(
            f"Nearest station: {dispatch.get('station')} (matched on "
            f"'{dispatch.get('matched_on')}')"
            + (f", ETA {eta} minutes" if eta else "")))
    elif dispatch.get('station'):
        pdf.multi_cell(0, 10, sanitize_text(
            f"Nearest station: {dispatch.get('station')} - no location identified "
            "in the call, so this is the default for this emergency type. No ETA."))

    if data.get('truncated'):
        pdf.multi_cell(0, 10, sanitize_text(
            f"PARTIAL ANALYSIS: only the first {data.get('analysed_duration_s')}s "
            f"of a {data.get('source_duration_s')}s recording was analysed."))
    
    # Emotional Analysis
    pdf.set_font("Arial", 'B', 14)
    pdf.multi_cell(0, 10, sanitize_text("\nEmotional Analysis\n"))
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, sanitize_text(f"Sentiment: {data['sentiment']['label']} (Confidence: {data['sentiment']['score']:.2f})"))
    pdf.multi_cell(0, 10, sanitize_text(f"Emotion: {data['emotion']['label']} (Confidence: {data['emotion']['score']:.2f})\n"))
    
    # Recommended Actions
    pdf.set_font("Arial", 'B', 14)
    pdf.multi_cell(0, 10, sanitize_text("\nRecommended Actions\n"))
    pdf.set_font("Arial", size=12)
    for suggestion in data['response']['suggestions']:
        pdf.multi_cell(0, 10, sanitize_text(f"• {suggestion}"))

    signals = data['response'].get('signals') or {}
    if signals:
        pdf.multi_cell(0, 10, sanitize_text(
            "\nSome actions above were added because these words appeared in the "
            "transcript. They are keyword matches, not judgements:"))
        for group, words in signals.items():
            pdf.multi_cell(0, 10, sanitize_text(f"• {group}: {', '.join(words)}"))
    
    # Required Resources
    pdf.multi_cell(0, 10, sanitize_text("\nRequired Resources:"))
    for resource in data['response']['required_resources']:
        pdf.multi_cell(0, 10, sanitize_text(f"• {resource.replace('_', ' ').title()}"))
    
    # Transcription
    pdf.set_font("Arial", 'B', 14)
    pdf.multi_cell(0, 10, sanitize_text("\nTranscription\n"))
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, sanitize_text(f"{data['transcription']}\n"))
    
    # Summary
    pdf.set_font("Arial", 'B', 14)
    pdf.multi_cell(0, 10, sanitize_text("\nSummary\n"))
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, sanitize_text(f"{data['summary']}\n"))
    
    # One file per report. This used to write a single shared
    # `processed/fir_report.pdf`, so two concurrent callers overwrote each
    # other and /download_fir served whichever finished last -- one caller
    # could download another caller's report. It also ignored PROCESSED_FOLDER.
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    out_path = os.path.join(PROCESSED_FOLDER, f"fir_{data['report_id']}.pdf")
    pdf.output(out_path)
    return out_path
