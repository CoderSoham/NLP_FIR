import os
import time
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
from utils.dispatch import get_dispatch_suggestion, load_stations
from utils.geocode import locate as geocode_location
from utils.signals import augment_actions
from utils.severity import severity_score, severity_label
from utils.summary import length_budget, is_informative
from utils.policy import local_analysis_wanted, model_summary
from utils.compare import compare_classifications
from utils.llm import extract_incident, get_backend as llm_get_backend
from utils import gpu
from utils.jobs import spawn as jobs_spawn
from utils.asr import transcribe_file as asr_transcribe_file
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


STAGE_WARNINGS = []


def _collect_incident(future, transcript, **context):
    """Result of the extraction stage, whether it was forked or not.

    Returns (record, meta) and never raises -- a failed enrichment must not
    fail a request that has already transcribed and classified a call.
    """
    if future is None:
        return extract_incident(transcript, **context)
    waited = time.monotonic()
    record, meta = future()
    meta = dict(meta or {})
    # How long the join actually blocked, as opposed to how long the call
    # took. The difference is what the local stages managed to hide.
    meta["waited_seconds"] = round(time.monotonic() - waited, 1)
    # Worth recording. It is the difference between a 220s request and an 18s
    # one, and the first thing to check if the two paths ever start
    # disagreeing more than they used to.
    meta["concurrent"] = True
    return record, meta


def note_stage_failure(stage, exc):
    """Record a degraded stage once, in a form a reader can act on.

    A substituted neutral default is indistinguishable from a genuine neutral
    reading, so every substitution is recorded rather than swallowed.
    """
    message = f"{stage}: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
    if message not in STAGE_WARNINGS:
        STAGE_WARNINGS.append(message)
    print(f"[degraded] {message}")
    return message


def _classifier_pipeline(task, model_id, **kwargs):
    """Build a text pipeline, forcing safetensors.

    transformers >= 4.57 refuses torch.load on torch < 2.6 (CVE-2025-32434).
    Both checkpoints ship safetensors as well as a .bin, but the pipeline
    helper resolves to the .bin and dies -- and passing
    model_kwargs={"use_safetensors": True} does not reach the loader.
    Building the model and tokenizer explicitly does.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, use_safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return pipeline(task, model=model, tokenizer=tokenizer,
                    device=pipeline_device(), framework="pt", **kwargs)


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
        SENTIMENT_ANALYZER = _classifier_pipeline(
            "sentiment-analysis",
            "finiteautomata/bertweet-base-sentiment-analysis")
    return SENTIMENT_ANALYZER

def load_emotion_detector():
    global EMOTION_DETECTOR
    gpu.acquire('nlp')
    if EMOTION_DETECTOR is None:
        EMOTION_DETECTOR = _classifier_pipeline(
            "text-classification",
            "j-hartmann/emotion-english-distilroberta-base")
    return EMOTION_DETECTOR

# Sanitize Unicode for fpdf
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
        note_stage_failure("emergency_type", e)
        return "unknown"

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

        # Two of the four severity terms come from these models; losing them
        # quietly means a score computed from half its inputs, with a neutral
        # default that reads exactly like a genuine neutral result.
        try:
            sentiment = load_sentiment_analyzer()(text[:128])[0]
        except Exception as exc:
            sentiment = {"label": "NEU", "score": 0.5}
            note_stage_failure("sentiment", exc)
        try:
            emotion = load_emotion_detector()(text[:128])[0]
        except Exception as exc:
            emotion = {"label": "neutral", "score": 0.5}
            note_stage_failure("emotion", exc)

        score, _parts = severity_score(
            distribution, sentiment, emotion, entities,
            emotion_min_confidence=EMOTION_MIN_CONFIDENCE)
        return severity_label(score)

    except Exception as e:
        note_stage_failure("severity", e)
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

def process_audio_file(input_path, output_folder, progress=None):
    """Process audio file and generate analysis.

    `progress` receives a short phrase per stage, written for a reader waiting
    on a page rather than as a stage identifier.
    """
    say = progress or (lambda _stage, **_kw: None)
    try:
        # Load audio file with specific parameters for Whisper
        say("Loading audio")
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
        STAGE_WARNINGS.clear()          # per request, not per process

        # Condition the audio before recognition. Conservative on purpose --
        # see utils.audio_clean for why aggressive denoising hurts ASR.
        say("Cleaning audio")
        audio, audio_report = clean_audio(audio, sr)

        # Generate visualizations from what was actually transcribed, not from
        # the raw file, so the plots describe the analysed signal.
        say("Drawing waveform and spectrum")
        plots = generate_visualizations(audio, sr, output_folder, report_id)

        # faster-whisper takes the array directly, so the intermediate WAV that
        # every request used to write and delete is gone.
        # transcribe_file, not the array-level call: it honours ASR_BACKEND and
        # the transcript cache. The pipeline called the local function directly,
        # so ASR_BACKEND was read by nothing that mattered.
        say("Transcribing the call")
        asr_result = asr_transcribe_file(input_path, audio=audio, sr=sr)
        transcription = asr_result["text"]
        lang = asr_result["language"]

        # If non-English, translate to English for the downstream stages.
        say(None, transcript=transcription,
            language=lang, analysed_seconds=round(analysed_duration, 1),
            truncated=truncated)

        translated_text = transcription
        if lang and lang != 'en':
            try:
                translated_text = asr_transcribe_file(
                    input_path, audio=audio, sr=sr, translate=True,
                    use_cache=False)["text"]
            except Exception:
                translated_text = transcription
        
        # The LLM stage is a network call when the backend is hosted, and the
        # stages below are GPU work. Run them at the same time. Serially this
        # was 66s of waiting on a socket while a 6GB card sat idle; the join
        # below is usually instant by the time the classifiers finish.
        #
        # A *local* backend is explicitly excluded -- it competes for the same
        # VRAM as the classifiers, and overlapping them is how this pipeline
        # spent a week out of memory.
        llm_future = None
        llm_backend = llm_get_backend()
        if llm_backend is not None and getattr(llm_backend, "is_hosted", False):
            llm_future = jobs_spawn(
                extract_incident, translated_text, truncated=truncated,
                source_seconds=source_duration,
                analysed_seconds=analysed_duration)

        # Named entities first -- severity weights them, so they have to exist
        # before it runs. Previously severity was called with a hardcoded []
        # and the entity term of its score was dead on every request.
        say("Extracting names, places and numbers")
        ner_model = load_ner_model()
        doc = ner_model(translated_text)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        if generate_entity_plot(entities, output_folder, report_id):
            plots.append('entities')
        say(None, entities=entities[:12])

        # Get emergency type and severity
        # The incident record is collected before the local classifiers, not
        # after them, because whether it succeeded decides whether they have
        # to run at all. Only spaCy overlaps the network call now, so this
        # join waits -- and then saves a minute and a quarter of BART.
        say("Reading the call for an incident record")
        llm_record, llm_meta = _collect_incident(
            llm_future, translated_text, truncated=truncated,
            source_seconds=source_duration, analysed_seconds=analysed_duration)

        second_opinion = local_analysis_wanted(llm_record)
        llm_meta = dict(llm_meta or {})
        llm_meta['second_opinion'] = second_opinion

        if second_opinion:
            say("Classifying the emergency")
            emergency_type = get_emergency_type(translated_text)
            severity = assess_severity(translated_text, entities)
        else:
            # The model already answered both, better and faster, and it was
            # measured doing so. Two BART-large passes to produce a worse
            # version of an answer already in hand is not a second opinion.
            emergency_type = (llm_record.get('incident_type') or 'unknown')
            severity = llm_record.get('severity') or 'low'

        response = get_emergency_response(emergency_type)
        response = augment_actions_from_transcript(translated_text, response)
        response_time = calculate_response_time(emergency_type, response['priority'])
        say(None, emergency_type=emergency_type, severity=severity,
            priority=response['priority'], response_time=response_time)

        # Summarise only when the model did not. Two summaries of one call is
        # not a second opinion either, and the report already suppressed this
        # one in favour of the model's -- so the pipeline was spending a
        # minute of BART on text nobody would read.
        summary = model_summary(llm_record)
        summarised_by = "model"
        if not summary:
            say("Summarising")
            summarised_by = "bart"
            summary = summarize_text_chunked(translated_text)
            if not is_summary_informative(summary, translated_text):
                # Fallback to T5; if still low-value, leave empty for UI to hide
                summary = summarize_text(translated_text)
                if not is_summary_informative(summary, translated_text):
                    summary = ""
        say(None, summary=summary, summarised_by=summarised_by)

        # Simple location extraction heuristic (reuses the doc parsed above)
        probable_location = None
        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC", "FAC"):
                probable_location = ent.text
                break

        # Three more local models, all display-only once the record exists.
        # Caller affect is a weak signal that feeds the severity score as one
        # term of four -- and when the model classified the call, that scorer
        # did not run, so these feed nothing at all. Together they are ~8s of
        # cold loading for a line in the PDF.
        #
        # They are reported as absent rather than as a neutral default. A
        # fabricated "NEU 0.50" is indistinguishable from a genuine neutral
        # reading, which is how two dead models went unnoticed for a week --
        # see note_stage_failure and the severity scorer.
        best_match = score = sentiment = emotion = None
        if second_opinion:
            embedder = load_embedder()
            with torch.no_grad():
                known_emb = get_known_embeddings()
                query_emb = embedder.encode([translated_text], convert_to_tensor=True)
                cos_scores = util.cos_sim(query_emb, known_emb)[0]
                top_idx = int(torch.argmax(cos_scores).item())
                best_match = known_commands[top_idx]
                score = float(cos_scores[top_idx].item())

            try:
                sentiment = load_sentiment_analyzer()(translated_text[:256])[0]
            except Exception as exc:
                note_stage_failure("sentiment", exc)
            try:
                emotion = load_emotion_detector()(translated_text[:256])[0]
            except Exception as exc:
                note_stage_failure("emotion", exc)
        
        # Generate PDF report
        # Dispatch suggestion
        # The model's location is used before spaCy's: it returns the address
        # the caller gave, where spaCy returns whichever GPE appeared first in
        # the transcript -- often a city mentioned in passing.
        dispatch_location = (llm_record or {}).get('location') or probable_location
        registry = load_stations()
        dispatch = get_dispatch_suggestion(
            emergency_type, dispatch_location, stations=registry,
            incident_coords=geocode_location(dispatch_location, registry))

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
            'summarised_by': summarised_by,
            'second_opinion': second_opinion,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'entities': entities,
            'plots': plots,
            'asr': {k: v for k, v in asr_result.items() if k != 'segments'},
            'asr_segments': asr_result['segments'],
            'audio_report': audio_report,
            'warnings': list(STAGE_WARNINGS),
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
        
        # The classical path stays the floor, never the fallback: both
        # results are kept so a disagreement between them can be surfaced
        # rather than resolved silently.
        data['llm'] = llm_record
        data['llm_meta'] = llm_meta
        data['disagreements'] = compare_classifications(
            data, llm_record, second_opinion=second_opinion)

        say("Writing the report")
        data['fir_pdf'] = generate_fir_pdf(data)

        return data
        
    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        raise  # Re-raise the exception to handle it in the Flask route

# A plot is at most ~1200px wide, so it cannot show more detail than that many
# columns however much audio sits behind it. Rendering a ten-minute recording at
# 16 kHz took 80 seconds -- 36% of the whole request, more than the language
# model -- nearly all of it in piptrack, which is expensive per frame.
PLOT_MAX_SECONDS = 180
PLOT_HOP_LENGTH = 1024
# A plot is about 1200 pixels wide and cannot show more columns than that, so
# there is nothing to gain from drawing at 16 kHz.
PLOT_SAMPLE_RATE = 8000
# The human speech fundamental. Male voices bottom out near 85 Hz, and nobody
# speaks above about 400 -- anything outside this is an octave error or noise.
PITCH_FMIN = 65.0
PITCH_FMAX = 400.0
# Voiced-frame gate, as a fraction of this recording's upper-quartile RMS. A
# fixed threshold would depend on the recording's gain.
VOICED_RMS_FRACTION = 0.35


def _downsample_for_plot(audio, sr):
    """Resample to a rate that still fills the plot, and cap the span drawn.

    `librosa.resample`, not `audio[::step]`. Naive decimation applies no
    anti-alias filter, so every component above the new Nyquist folded back
    into the band -- the MFCC and the pitch chart were both drawn from a
    signal containing energy that is not in the recording.
    """
    target_sr = min(sr, PLOT_SAMPLE_RATE)
    if target_sr < sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr,
                                 res_type="polyphase")
    limit = PLOT_MAX_SECONDS * target_sr
    return audio[:limit], target_sr, len(audio) > limit


def generate_visualizations(audio, sr, output_folder, report_id):
    """Signal diagnostics. Returns the plot kinds written.

    These describe the *audio*, not the incident -- they are diagnostics for
    someone debugging a bad transcription, which is why the result page keeps
    them collapsed. That does not excuse them from being correct: the pitch
    chart used to plot the strongest spectral peak up to 2 kHz and label it
    pitch, when speech fundamentals live below 400 Hz, and the MFCC colourbar
    was labelled in decibels, which cepstral coefficients are not.
    """
    os.makedirs(output_folder, exist_ok=True)
    audio, sr, clipped = _downsample_for_plot(audio, sr)
    suffix = f" (first {PLOT_MAX_SECONDS // 60} minutes)" if clipped else ""

    plt.figure(figsize=(12, 3.2))
    librosa.display.waveshow(audio, sr=sr)
    plt.title("Waveform" + suffix)
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(plot_path(output_folder, 'waveform', report_id))
    plt.close()

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13,
                                 hop_length=PLOT_HOP_LENGTH)
    # Coefficient 0 is overall energy and is an order of magnitude larger than
    # the rest, so including it in the colour scale flattened every other row
    # into one indistinguishable band -- which is exactly how this chart used
    # to look. The timbre information is in 1..12.
    detail = mfccs[1:]
    span = float(np.percentile(np.abs(detail), 99)) or 1.0
    plt.figure(figsize=(12, 3.2))
    librosa.display.specshow(detail, x_axis='time', sr=sr,
                             hop_length=PLOT_HOP_LENGTH,
                             vmin=-span, vmax=span, cmap='coolwarm')
    # No unit. MFCCs are unitless cepstral coefficients; the old colourbar
    # said "dB", which was simply false.
    plt.colorbar(label="coefficient value")
    plt.yticks(np.arange(detail.shape[0]), np.arange(1, detail.shape[0] + 1))
    plt.ylabel("MFCC coefficient")
    plt.title("MFCC 1-12" + suffix)
    plt.tight_layout()
    plt.savefig(plot_path(output_folder, 'mfcc', report_id))
    plt.close()

    plt.figure(figsize=(12, 3.2))
    times, contour = _pitch_contour(audio, sr)
    if contour is not None:
        plt.plot(times, contour, linewidth=0.9)
        plt.ylim(PITCH_FMIN, PITCH_FMAX)
    else:
        plt.text(0.5, 0.5, "No voiced speech detected",
                 ha="center", va="center", transform=plt.gca().transAxes)
    plt.ylabel("Fundamental frequency (Hz)"); plt.xlabel("Time (s)")
    plt.title("Pitch, voiced frames only" + suffix)
    plt.tight_layout()
    plt.savefig(plot_path(output_folder, 'pitch', report_id))
    plt.close()

    return ['waveform', 'mfcc', 'pitch']


def _pitch_contour(audio, sr):
    """Fundamental frequency over time, or (times, None) if nothing is voiced.

    `yin` bounded to the human speech range, not `piptrack`. piptrack's argmax
    over magnitude returns the strongest spectral peak, which for speech is
    usually a formant or a harmonic rather than F0 -- with fmax=2000 the old
    chart plotted values up to 1850 Hz and called them pitch. No adult speaks
    above about 400 Hz.

    Unvoiced frames are dropped rather than drawn. yin returns an estimate for
    every frame including silence, so without a gate the chart showed a busy
    contour during passages where nobody was talking.
    """
    frame_length = 1024
    try:
        f0 = librosa.yin(audio, fmin=PITCH_FMIN, fmax=PITCH_FMAX, sr=sr,
                         frame_length=frame_length, hop_length=PLOT_HOP_LENGTH)
    except Exception as exc:
        note_stage_failure("pitch", exc)
        return np.array([]), None

    rms = librosa.feature.rms(y=audio, frame_length=frame_length,
                              hop_length=PLOT_HOP_LENGTH)[0][:len(f0)]
    f0 = f0[:len(rms)]
    # A fixed threshold would depend on the recording's gain, so the gate is
    # relative to the loud parts of this recording.
    gate = float(np.percentile(rms, 75)) * VOICED_RMS_FRACTION
    voiced = (rms > gate) & (f0 > PITCH_FMIN) & (f0 < PITCH_FMAX)
    if not voiced.any():
        return np.array([]), None

    contour = np.where(voiced, f0, np.nan)
    times = librosa.frames_to_time(np.arange(len(contour)), sr=sr,
                                   hop_length=PLOT_HOP_LENGTH)
    return times, contour


def generate_fir_pdf(data):
    """Write the incident report. The document lives in utils.fir_pdf.

    Kept as a thin wrapper because app.py, the tests and the API route all
    import this name, and the layout has no business in a module that also
    loads eight models.
    """
    from utils import fir_pdf
    return fir_pdf.generate(data, PROCESSED_FOLDER)
