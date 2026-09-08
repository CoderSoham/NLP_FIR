"""Plot rendering. Deliberately free of torch, whisper and spacy imports.

Split out so it can be imported -- and tested -- without loading 4.5 GB of
model weights. `generate_visualizations` stays in audio_utils because it needs
librosa; everything here needs only matplotlib.
"""
import os
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_path(output_folder, kind, report_id):
    """Where one plot for one request lives.

    Every artefact is keyed by report_id and kept out of the public static
    mount. These used to be fixed names -- `waveform.png` and friends -- written
    into a publicly served directory, so two concurrent uploads overwrote each
    other and the result page showed the other caller's audio. Anyone could also
    fetch the path directly and see the most recently processed call.
    """
    return os.path.join(output_folder, f"{kind}_{report_id}.png")

def generate_entity_plot(entities, output_folder, report_id):
    """Plot the distribution of NER labels. Returns True if a plot was written.

    Kept separate from generate_visualizations because that runs on the raw
    audio at step 2, before NER exists. This needs the entities, so it runs
    after them.

    Returns False when there are no entities rather than writing an empty
    figure: a call in which spaCy recognises nothing is normal, not an error,
    and the template renders the image only when this returned True. The
    previous version of index.html referenced this plot unconditionally and
    nothing ever generated it, so every result page carried a broken image.
    """
    labels = [e['label'] for e in entities]
    if not labels:
        return False

    os.makedirs(output_folder, exist_ok=True)

    # Counted and sorted rather than plt.hist over raw strings, which bins
    # categorical data by insertion order and mislabels the axis.
    ordered = Counter(labels).most_common()
    names = [n for n, _ in ordered]
    values = [v for _, v in ordered]

    plt.figure(figsize=(12, 4))
    plt.bar(names, values)
    plt.title('Named Entity Distribution')
    plt.ylabel('Count')
    # counts are integers; the default locator emits 0.5 steps on small values
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plot_path(output_folder, 'entities', report_id))
    plt.close()
    return True

