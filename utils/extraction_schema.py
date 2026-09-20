"""The structured incident record the LLM stage produces.

One schema, used three ways: as the JSON Schema for Anthropic's structured
outputs, as the shape described to a local model in its prompt, and as the
validator for whatever comes back. Keeping one definition is what stops the
three drifting apart.

Every field is nullable. A 911 transcript frequently does not contain the
address, the caller's name or the number of people hurt, and a model that
invents them is worse than one that returns null -- which is the failure this
whole pipeline keeps re-learning.
"""

INCIDENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "incident_type", "severity", "is_in_progress", "location",
        "callback_number", "caller_name", "people_involved", "injuries",
        "weapons", "hazards", "vehicles", "services_needed",
        "immediate_actions", "summary", "uncertainties",
    ],
    "properties": {
        "incident_type": {
            "type": "string",
            "enum": ["medical", "fire", "police", "accident", "other", "unknown"],
            "description": "The primary emergency category.",
        },
        "severity": {
            "type": "string",
            "enum": ["critical", "high", "medium", "low", "unknown"],
            "description": (
                "Immediate risk to life. critical = someone is dying now "
                "(not breathing, severe bleeding, active violence). "
                "low = no injury and no ongoing threat."
            ),
        },
        "is_in_progress": {
            "type": ["boolean", "null"],
            "description": "Is the incident still happening, or already over?",
        },
        "location": {
            "type": ["string", "null"],
            "description": (
                "The incident address as stated by the caller, verbatim. "
                "Null if never stated. Do not infer or complete an address."
            ),
        },
        "callback_number": {"type": ["string", "null"]},
        "caller_name": {"type": ["string", "null"]},
        "people_involved": {
            "type": ["integer", "null"],
            "description": "Count of people at the scene, if stated or clearly implied.",
        },
        "injuries": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Injuries described. Empty if none mentioned.",
        },
        "weapons": {
            "type": "array",
            "description": (
                "Weapons actually present, each with the words from the "
                "transcript that establish it. A figure of speech, a joke, a "
                "hypothetical or a threat that was not carried out is NOT a "
                "weapon. Body parts used in a fight are NOT weapons. Empty if "
                "none."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["item", "quote"],
                "properties": {
                    "item": {"type": "string",
                             "description": "The weapon, e.g. 'handgun', 'knife'."},
                    "quote": {
                        "type": "string",
                        "description": (
                            "The exact words from the transcript establishing "
                            "it, copied verbatim. If you cannot quote it, do "
                            "not list it."
                        ),
                    },
                },
            },
        },
        "hazards": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Fire, gas, water, electrical, structural, animal.",
        },
        "vehicles": {"type": "array", "items": {"type": "string"}},
        "services_needed": {
            "type": "array",
            "items": {"type": "string", "enum": ["police", "fire", "ems"]},
        },
        "immediate_actions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete dispatcher actions, most urgent first.",
        },
        "summary": {
            "type": "string",
            "description": "Two or three sentences a dispatcher could read aloud.",
        },
        "uncertainties": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "What the transcript does not establish, or where it is garbled. "
                "This field is required, not optional -- an extraction that "
                "claims total confidence in a noisy phone recording is wrong."
            ),
        },
    },
}

SYSTEM_PROMPT = """\
You extract structured incident records from transcripts of emergency calls.

The transcripts come from automatic speech recognition of telephone audio. They \
contain recognition errors, especially in names, street names and numbers. Some \
are partial recordings that stop mid-call.

Rules:

- Extract only what the transcript states. If a field is not established, return \
null or an empty list. Never infer a plausible address, name or phone number.
- Distinguish speakers. A 911 transcript contains a dispatcher and a caller, \
usually unlabelled. Something the dispatcher says is not a report by the caller.
- Read for intent, not keywords. "Did you want us to come over and shoot her?" is \
sarcasm, not a firearm. "He's having a cardiac arrest" is medical, not a vehicle.
- A weapon is an instrument that is present at the scene. Hands, feet and fists \
used in a fight are not weapons. Something mentioned only as a joke, a threat \
that was not acted on, or a hypothetical is not present. **Every weapon you \
list must be supported by words you can quote from the transcript** -- if you \
cannot quote it, do not list it.
- Where the transcript is garbled, say so in `uncertainties` rather than guessing. \
If a location sounds like ASR noise rather than a real place, treat it as absent \
and record that.
- Severity means immediate risk to life, not how upset anyone sounds.

Return only the structured record."""


def build_user_prompt(transcript, truncated=False, source_seconds=None,
                      analysed_seconds=None):
    parts = []
    if truncated:
        # Durations may be absent -- a caller can know the transcript is partial
        # without knowing the original length. Formatting None crashed here.
        if source_seconds and analysed_seconds:
            parts.append(
                f"NOTE: this is the first {analysed_seconds:.0f} seconds of a "
                f"{source_seconds:.0f}-second recording. The call continues "
                "beyond the transcript, so the outcome may not appear. Reflect "
                "that in `uncertainties`.")
        else:
            parts.append(
                "NOTE: this transcript is only the beginning of a longer "
                "recording. The call continues beyond it, so the outcome may "
                "not appear. Reflect that in `uncertainties`.")
    parts.append("Transcript:\n\n" + (transcript or "").strip())
    return "\n\n".join(parts)


def compact_field_spec():
    """A terse field list for local models, instead of the full JSON Schema.

    Serialising the schema as JSON costs ~815 tokens, and prompt length is not
    free on a local model: `generate()` computes logits for every prompt
    position on the first forward pass, so with Qwen2's 151,936-token
    vocabulary a 1797-token prompt needs a single 1.03 GiB allocation. That
    does not fit beside the weights on a 6 GB card.

    **Written by hand, not generated.** A generated version that emitted only
    names and types came to 123 tokens and dropped the score from 11/18 to
    6/18 -- the field descriptions were carrying the behaviour ("critical means
    someone is dying now", "do not infer an address", "uncertainties is
    required"), and without them the model stopped filling `uncertainties` at
    all. The saving is in dropping JSON Schema boilerplate, not the semantics.

    The hosted backend keeps the full schema: structured outputs enforce it
    server-side and the length costs nothing there.
    """
    return """\
  incident_type: medical|fire|police|accident|other|unknown
  severity: critical|high|medium|low|unknown
      critical = someone is dying now (not breathing, severe bleeding, active
      violence). low = no injury and no ongoing threat. Severity is risk to
      life, not how upset anyone sounds.
  is_in_progress: true|false|null - still happening, or already over?
  location: string|null
      The address as the caller states it, verbatim. null if never stated.
      Never infer or complete an address. A short span, not a paragraph.
  callback_number: string|null
  caller_name: string|null
  people_involved: integer|null
  injuries: [string] - injuries described; [] if none
  weapons: [{item, quote}]
      Only weapons actually present. Hands, feet and fists used in a fight are
      NOT weapons. A joke, a hypothetical, or a threat not acted on is NOT a
      weapon. `quote` must be copied verbatim from the transcript; if you
      cannot quote it, do not list it. [] if none.
  hazards: [string] - fire, gas, water, electrical, structural, animal
  vehicles: [string]
  services_needed: [police|fire|ems] - these three words only
  immediate_actions: [string] - concrete dispatcher actions, most urgent first
  summary: string - two or three sentences a dispatcher could read aloud
  uncertainties: [string]
      REQUIRED, never empty. What the transcript does not establish, or where
      it is garbled. An extraction that claims total confidence in a noisy
      phone recording is wrong."""


# Models reach for the vocabulary a dispatcher would use rather than the three
# words the schema allows. Observed across runs: "emt", "EMS", "ambulance",
# "emergencies", "paramedics", "fire department". Rejecting these as invalid is
# technically correct and practically wrong -- the model identified the right
# service and spelled it differently.
SERVICE_ALIASES = {
    "ems": "ems", "emt": "ems", "emts": "ems", "ambulance": "ems",
    "paramedic": "ems", "paramedics": "ems", "medical": "ems",
    "emergency medical": "ems", "emergency medical services": "ems",
    "police": "police", "law enforcement": "police", "officers": "police",
    "police department": "police", "sheriff": "police", "pd": "police",
    "fire": "fire", "fire department": "fire", "firefighters": "fire",
    "fire brigade": "fire", "fd": "fire",
}


def normalise_service(value):
    """Map a service name onto the schema's enum, or None if it is not one.

    Deliberately not a fuzzy match: "emergencies" is not a service and should
    still be rejected, because it means the model did not decide.
    """
    key = str(value or "").strip().lower().rstrip(".")
    return SERVICE_ALIASES.get(key)
