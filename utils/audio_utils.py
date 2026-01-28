import os
import whisper
import torch
import librosa
import soundfile as sf
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive 'Agg'
import matplotlib.pyplot as plt
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
from config import PROCESSED_FOLDER, STATIC_PLOTS_FOLDER, WHISPER_MODEL_NAME
from typing import Optional, Tuple

# Initialize models as None for lazy loading
whisper_model = None
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

def load_whisper_model():
    global whisper_model
    if whisper_model is None:
        # Force CPU usage and set specific parameters
        whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device="cpu")
        # Set model parameters for better stability
        whisper_model.eval()  # Set to evaluation mode
    return whisper_model

def load_embedder():
    global embedder
    if embedder is None:
        embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    return embedder

def load_nlp():
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_sm", disable=['parser', 'textcat'])
    return nlp

def load_t5():
    global t5_tokenizer, t5_model
    if t5_tokenizer is None or t5_model is None:
        t5_model_name = "t5-small"
        t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        t5_model.eval()  # Set to evaluation mode
    return t5_tokenizer, t5_model

def load_emergency_classifier():
    global EMERGENCY_CLASSIFIER
    if EMERGENCY_CLASSIFIER is None:
        EMERGENCY_CLASSIFIER = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,  # Force CPU usage
            framework="pt"  # Use PyTorch backend
        )
    return EMERGENCY_CLASSIFIER

def load_severity_classifier():
    global SEVERITY_CLASSIFIER
    if SEVERITY_CLASSIFIER is None:
        SEVERITY_CLASSIFIER = pipeline(
            "zero-shot-classification",
            model="microsoft/deberta-v3-base",
            device=-1,  # Force CPU usage
            framework="pt"  # Use PyTorch backend
        )
    return SEVERITY_CLASSIFIER

def load_ner_model():
    global NER_MODEL
    if NER_MODEL is None:
        NER_MODEL = spacy.load("en_core_web_sm", disable=['parser', 'textcat'])
    return NER_MODEL

def load_sentence_model():
    global SENTENCE_MODEL
    if SENTENCE_MODEL is None:
        SENTENCE_MODEL = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    return SENTENCE_MODEL

def load_summarizer():
    global SUMMARIZER
    if SUMMARIZER is None:
        SUMMARIZER = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1,  # Force CPU usage
            framework="pt"  # Use PyTorch backend
        )
    return SUMMARIZER

def load_sentiment_analyzer():
    global SENTIMENT_ANALYZER
    if SENTIMENT_ANALYZER is None:
        SENTIMENT_ANALYZER = pipeline(
            "sentiment-analysis",
            model="finiteautomata/bertweet-base-sentiment-analysis",
            device=-1,  # Force CPU usage
            framework="pt"  # Use PyTorch backend
        )
    return SENTIMENT_ANALYZER

def load_emotion_detector():
    global EMOTION_DETECTOR
    if EMOTION_DETECTOR is None:
        EMOTION_DETECTOR = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=-1,  # Force CPU usage
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
        summary_ids = t5_model.generate(
            input_ids,
            max_length=200,  # Increased max length for more detailed summary
            min_length=30,   # Increased min length to ensure sufficient detail
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
                out = summarizer(ch, max_length=120, min_length=40, do_sample=False)[0]['summary_text']
                summaries.append(out)
    except Exception:
        # If BART errors, fall back to existing T5 summarizer
        return summarize_text(text)
    # Combine summaries
    combined = " ".join(summaries)
    # Final pass to tighten
    try:
        with torch.inference_mode():
            final = summarizer(combined, max_length=160, min_length=50, do_sample=False)[0]['summary_text']
            return clean_summary(final)
    except Exception:
        return clean_summary(combined)

def is_summary_informative(summary: str, source: str) -> bool:
    if not summary:
        return False
    if len(summary.split()) < 20:
        return False
    keywords = [
        'address', 'location', 'street', 'avenue', 'boulevard',
        'suspect', 'vehicle', 'truck', 'car', 'license', 'plate',
        'weapon', 'gun', 'shot', 'fire', 'smoke', 'injury', 'bleeding'
    ]
    s = summary.lower()
    if any(k in s for k in keywords):
        return True
    # If not in summary, check source contains critical signals; if yes, relax threshold
    src = (source or '').lower()
    if any(k in src for k in ['shot', 'gun', 'fire', 'smoke', 'unconscious', 'not breathing']):
        return len(summary.split()) >= 15
    return False

def augment_actions_from_transcript(transcript: str, base_response: dict) -> dict:
    """Augment recommended actions based on high-signal phrases in transcript."""
    t = (transcript or '').lower()
    suggestions = list(base_response.get('suggestions', []))
    # Violence/domestic
    if any(k in t for k in ['domestic', 'ex', 'restraining order', 'stalking', 'threaten', 'violent']):
        suggestions += [
            'Dispatch police to secure the scene',
            'Check for restraining order or prior incidents',
            'Advise caller to stay in a safe locked room'
        ]
    # Weapons/shots
    if any(k in t for k in ['shot', 'shoot', 'gun', 'gunfire']):
        suggestions += [
            'Issue officer safety advisory (possible firearm)',
            'Request additional police units'
        ]
    # Medical critical
    if any(k in t for k in ['not breathing', 'unconscious', 'severe bleeding', 'cpr']):
        suggestions += [
            'Provide pre-arrival instructions (CPR/bleeding control)',
            'Dispatch ALS ambulance'
        ]
    # Fire/hazmat
    if any(k in t for k in ['smoke', 'flames', 'burning', 'gas leak', 'explosion']):
        suggestions += [
            'Shut off gas/electric if safe',
            'Keep bystanders clear and evacuate adjacent units'
        ]
    # Vehicle/traffic
    if any(k in t for k in ['vehicle', 'truck', 'car', 'crash', 'collision']):
        suggestions += [
            'Notify traffic control for scene safety'
        ]
    # De-duplicate preserving order
    seen = set()
    dedup = []
    for s in suggestions:
        if s not in seen:
            dedup.append(s)
            seen.add(s)
    base_response['suggestions'] = dedup
    return base_response

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

# Simple station registry for dispatch suggestion (placeholder)
STATIONS = {
    'medical': [
        {'name': 'Central Hospital', 'area_keywords': ['downtown', 'central', 'hospital'], 'base_eta_min': 6, 'units': ['ambulance']},
        {'name': 'North Clinic', 'area_keywords': ['north', 'uptown'], 'base_eta_min': 10, 'units': ['ambulance']},
    ],
    'fire': [
        {'name': 'Station 1', 'area_keywords': ['downtown', 'central'], 'base_eta_min': 5, 'units': ['fire_truck']},
        {'name': 'Station 7', 'area_keywords': ['east', 'industrial'], 'base_eta_min': 9, 'units': ['fire_truck']},
    ],
    'police': [
        {'name': 'Precinct A', 'area_keywords': ['downtown', 'central'], 'base_eta_min': 4, 'units': ['police_unit']},
        {'name': 'Precinct B', 'area_keywords': ['west', 'suburb'], 'base_eta_min': 8, 'units': ['police_unit']},
    ],
    'accident': [
        {'name': 'Traffic Response Team', 'area_keywords': ['highway', 'bridge', 'junction'], 'base_eta_min': 7, 'units': ['traffic_unit', 'ambulance']},
    ]
}

def get_dispatch_suggestion(emergency_type, probable_location):
    """Generate a naive dispatch suggestion based on type and a keyword match on location."""
    et = emergency_type if emergency_type in STATIONS else 'accident'
    stations = STATIONS.get(et, [])
    if not stations:
        return {
            'station': 'Unknown',
            'eta_min': 15,
            'units': ['emergency_team'],
        }
    loc = (probable_location or '').lower()
    best = None
    for s in stations:
        if any(k in loc for k in s['area_keywords']):
            best = s
            break
    if best is None:
        best = stations[0]
    return {
        'station': best['name'],
        'eta_min': best['base_eta_min'],
        'units': best['units']
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
    """Optimized severity assessment using multiple ML models"""
    if not text or len(text.strip()) < 3:
        return "low"  # Return low severity for empty or very short text
    
    # Truncate text to prevent token length issues
    text = text[:500]  # Limit to 500 characters
        
    try:
        # Get base severity classification
        classifier = load_severity_classifier()
        severity_result = classifier(text, candidate_labels=["high", "medium", "low"])
        
        # Get sentiment and emotion
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
        
        # Calculate severity score
        severity_score = 0
        
        # Base severity from classifier
        severity_score += severity_result['scores'][0] * 0.4
        
        # Adjust based on sentiment
        if sentiment['label'] == 'NEG':
            severity_score += sentiment['score'] * 0.2
        
        # Adjust based on emotion
        if emotion['label'] in ['fear', 'anxiety']:
            severity_score += emotion['score'] * 0.2
        
        # Adjust based on entities
        entity_count = len(entities)
        severity_score += min(entity_count * 0.1, 0.2)
        
        # Determine final severity
        if severity_score > 0.7:
            return "high"
        elif severity_score > 0.4:
            return "medium"
        else:
            return "low"
            
    except Exception as e:
        print(f"Error in severity assessment: {str(e)}")
        return "low"  # Return low severity if assessment fails

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

        # Trim excessively long audio to bound processing time (e.g., 2 minutes)
        max_seconds = 120
        if len(audio) > sr * max_seconds:
            audio = audio[: sr * max_seconds]
        
        # Generate visualizations
        generate_visualizations(audio, sr, output_folder)
        
        # Convert to WAV if needed
        if not input_path.endswith('.wav'):
            output_path = os.path.join(output_folder, 'temp.wav')
            sf.write(output_path, audio, sr)
        else:
            output_path = input_path
        
        # Load and use Whisper model with specific parameters
        whisper_model = load_whisper_model()
        with torch.inference_mode():
            # First pass: detect language without forcing
            result = whisper_model.transcribe(
                output_path,
                fp16=False,
                task='transcribe'
            )
        transcription = result.get("text", "")

        # Language information
        lang = result.get('language', 'en') if isinstance(result, dict) else 'en'

        # If non-English, attempt translation to English for downstream analysis
        translated_text = transcription
        if lang and lang != 'en':
            with torch.inference_mode():
                try:
                    tr_result = whisper_model.transcribe(
                        output_path,
                        fp16=False,
                        task='translate'
                    )
                    translated_text = tr_result.get("text", transcription)
                except Exception:
                    translated_text = transcription
        
        # Get emergency type and severity
        emergency_type = get_emergency_type(translated_text)
        severity = assess_severity(translated_text, [])
        
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

        # Compute NLP extras for UI and PDF
        ner_model = load_ner_model()
        doc = ner_model(translated_text)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

        # Simple location extraction heuristic
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
            'best_match': best_match,
            'score': score,
            'sentiment': sentiment,
            'emotion': emotion,
            'emergency_response': response,
            'probable_location': probable_location,
            'dispatch': dispatch
        }
        
        generate_fir_pdf(data)
        
        return data
        
    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        raise  # Re-raise the exception to handle it in the Flask route

def generate_visualizations(audio, sr, output_folder):
    """Generate audio visualizations"""
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(STATIC_PLOTS_FOLDER, exist_ok=True)
    
    # Waveform plot
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title('Waveform')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'waveform.png'))
    plt.savefig(os.path.join(STATIC_PLOTS_FOLDER, 'waveform.png'))
    plt.close()
    
    # MFCC plot
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'mfcc.png'))
    plt.savefig(os.path.join(STATIC_PLOTS_FOLDER, 'mfcc.png'))
    plt.close()
    
    # Pitch plot
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    plt.figure(figsize=(12, 4))
    plt.plot(pitches)
    plt.title('Pitch')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'pitch.png'))
    plt.savefig(os.path.join(STATIC_PLOTS_FOLDER, 'pitch.png'))
    plt.close()

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
    pdf.multi_cell(0, 10, sanitize_text(f"Estimated Response Time: {data['response']['estimated_response_time']} minutes\n"))
    
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
    
    pdf.output(os.path.join("processed", "fir_report.pdf"))
