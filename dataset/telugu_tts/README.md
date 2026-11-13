# Telugu TTS Dataset Recording & Preprocessing Guide

**Dataset:** Single-Speaker Domain-Adaptive Telugu Speech Dataset for IndicF5 LoRA Fine-Tuning  
**Target Duration:** Pilot: 15–20 minutes (~120 clean utterances) | Full: 1–3 hours  
**Audio Target Specification:** 24,000 Hz, Mono, 16-bit PCM WAV ($0.3\text{ s} \le \text{duration} \le 30.0\text{ s}$)

---

## 1. Directory Structure

```text
dataset/telugu_tts/
├── sentence_manifest.json    # Starter sentence templates (120 utterances across 4 categories)
├── raw/                      # User-recorded raw audio files (e.g. sample_0001.wav)
├── processed/
│   ├── wavs/                 # Normalized 24kHz mono 16-bit PCM WAV files
│   ├── metadata.csv          # Complete dataset metadata (audio_file|text)
│   ├── train.csv             # 80% Training split (audio_file|text)
│   ├── val.csv               # 10% Validation split (audio_file|text)
│   ├── test.csv              # 10% Unseen Evaluation split (audio_file|text)
│   └── dataset_manifest.json # Diagnostic manifest with durations & categories
├── scripts/
│   ├── preprocess.py         # Audio normalization, silence trimming, duration filtering
│   ├── split_dataset.py      # Stratified deterministic train/val/test splitter
│   └── validate_dataset.py   # Dataset quality report & IndicF5 CustomDataset dry-run
└── README.md
```

---

## 2. Practical Recording Guidelines

For clean acoustic LoRA adaptation on `ai4bharat/IndicF5`, follow these recording protocols:

### A. Acoustic Environment
- **Quiet Room:** Record in a quiet indoor room away from street traffic, fans, AC units, and open windows.
- **Dampened Reverb:** Soft furnishings (carpets, curtains, sofas) help absorb room acoustic reflections.

### B. Hardware & Microphone Setup
- **Single Speaker:** Use the **same speaker** across the entire dataset to ensure vocal consistency.
- **Microphone Distance:** Maintain a steady distance of **15 to 20 cm** (approx. 6–8 inches) from the microphone.
- **Pop Filter:** Use a pop filter or slight off-axis angle to avoid plosive air bursts ("p", "t", "k").
- **Consistent Gain:** Set the recording input gain so peak volumes reach **-6 dB to -3 dB** (avoid digital clipping at 0 dB).

### C. Speaking Style & Delivery
- **Natural Pace:** Speak at a natural conversational speed (approx. 3 to 4 words per second).
- **Natural Pauses:** Leave a small 200–300 ms natural pause at the beginning and end of each sentence.
- **Pronunciation:** Speak clear, authentic Telugu. For code-mixed English terms (e.g., *database*, *embedding*, *API*), use standard natural Indian English pronunciation.
- **No Over-Enunciation:** Avoid robotic, forced, or hyper-articulated reading.
- **Retakes:** If you cough, stumble, or hesitate, re-record that sentence.

### D. File Naming
Save each recording with the corresponding ID from `sentence_manifest.json`:
- `raw/sample_0001.wav`
- `raw/sample_0002.wav`
- ... up to `raw/sample_0120.wav`

Raw audio can be in WAV, FLAC, M4A, OGG, or MP3 format at 44.1kHz or 48kHz; the preprocessing script automatically normalizes them to 24kHz mono 16-bit PCM.

---

## 3. Dataset Preprocessing & Validation Pipeline

Once raw recordings are placed in `dataset/telugu_tts/raw/`, run the pipeline in order:

### Step 1: Preprocess Audio & Generate Metadata
```bash
python dataset/telugu_tts/scripts/preprocess.py
```
- Converts raw recordings to mono 24,000 Hz 16-bit PCM WAV in `processed/wavs/`.
- Validates durations ($0.3\text{ s} \le \text{duration} \le 30.0\text{ s}$).
- Generates `processed/metadata.csv` (`audio_file|text` format).

### Step 2: Stratified Deterministic Split & Leakage Prevention
```bash
python dataset/telugu_tts/scripts/split_dataset.py --seed 42 --train 0.8 --val 0.1 --test 0.1
```
- Creates `train.csv`, `val.csv`, `test.csv`.
- Guarantees zero leakage ($\text{train} \cap \text{val} = \emptyset$, $\text{train} \cap \text{test} = \emptyset$, $\text{val} \cap \text{test} = \emptyset$).
- Stratifies representation across all 4 categories.

### Step 3: Dataset Quality Inspection & IndicF5 Dry-Run
```bash
python dataset/telugu_tts/scripts/validate_dataset.py
```
- Verifies sample rates, channels, durations, and metadata integrity.
- Runs dry-run test loading through `f5_tts.model.dataset.CustomDataset` and `collate_fn`.
