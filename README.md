# Stimulus Generation Pipeline (Typecast)

Python toolkit for building dichotic listening stimuli: voice selection, text generation, TTS, duration control, day/block packaging, and targeted regeneration.

## Important: always review generated outputs

Every generation step can produce imperfect text or audio.  
**Manual QA is required before using any output in experiments or sharing.**

Do not skip review after:

- topic / text generation
- TTS synthesis
- stretch / mixture assembly
- any regeneration or overwrite step

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export TYPECAST_API_TOKEN='your-typecast-api-key'
export OPENAI_API_KEY='your-openai-api-key'   # for topic/text/regeneration helpers
```

## Directory layout

```text
.
├── *.py                       # pipeline scripts
├── data/
│   ├── reference/             # voice metadata (fixed inputs)
│   ├── reference_eng/         # English day/block CSVs (unified)
│   └── generated/             # intermediate pipeline CSVs
└── outputs/                   # audio products (created locally; not shipped)
    ├── precue/ / precue_eng/
    ├── stimulus/ / stimulus_eng/
    └── stimulus_eng_final/
```

### Bundled data (no audio)

| Path | Contents |
|------|----------|
| `data/reference/` | `typecast_ko_standard_duration.csv/.json`, `typecast_voice_info.csv` |
| `data/reference_eng/` | `day1block*.csv` + `day2block*.csv` + `day3block*.csv` |
| `data/generated/` | `voice_info`, distances, `pairs`, topics, `text`, `text_aligned`, … |

English day sources used for this package:

- **day1** ← original `reference_eng`
- **day2** ← `reference_eng_d2`
- **day3** ← `reference_eng_d3`

All nine files live under a single `data/reference_eng/` folder so scripts can use one `--reference-dir` / `--input-dir`.

Precue and speech WAVs are **not** included; generate them locally under `outputs/`.

---

## A. Korean / shared upstream pipeline

Used to build the speaker pool, pair list, and (optionally) Korean text assets.

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1. Voice list | `typecast_voices.py` | Typecast API | `data/reference/typecast_ko_standard_duration.csv` |
| 2. Precue (KO) | `generate_pre_cue.py` | voice CSV | `outputs/precue/{name}_cue.wav` |
| 3. Speaking rate | `cal_velocity.py` | precue + `typecast_voice_info.csv` | `data/generated/voice_info.csv` |
| 4. Embeddings | `extract_embedding.py` | precue WAV | `outputs/precue/embeddings/` |
| 5. Distances | `cal_distance.py` | embeddings | `data/generated/pairwise_distances.csv` |
| 6. Gender split | `split_gender.py` | distances + voice_info | `MM/FF/MF_distance.csv` |
| 7. Pair sample | `sample_pairs.py` | gender distance CSVs | `data/generated/pairs.csv` |
| 8. Topics | `generate_topic.py` | OpenAI | `data/generated/topics.csv` |
| 9. Text + quiz | `generate_text.py` | topics CSV | `data/generated/text.csv` |
| 10. Assign speakers | `make_text_assigned.py` | pairs + text + voice_info | `data/generated/text_aligned.csv` |
| 11. TTS | `generate_speech.py` | text_aligned | `outputs/stimulus/` |
| 12. Stretch to 30 s | `stretch_audio.py` | stimulus WAV | `outputs/stimulus_stretched/` |

Example (voice list):

```bash
python typecast_voices.py --api-key "$TYPECAST_API_TOKEN"
```

Example (English precue, separate from Korean cue text):

```bash
python generate_pre_cue_eng.py
# default text: "Please focus on this voice"
# default out:  outputs/precue_eng/
```

---

## B. English day/block stimulus creation

Target layout for each `day{i}block{j}` (30 stimuli):

- stimuli **1–15**: base target / interference pairing  
- stimuli **16–30**: speaker roles switched relative to 1–15  
- final pack:
  - `target/`: 4 s padded precue + 30 s speech  
  - `interference/`: 4 s zero pad + paired speech  
  - `mixture/`: target + interference

### B1. Reference CSVs

This package already includes reviewed English tables under `data/reference_eng/`:

- file name: `day{i}block{j}.csv`
- required columns include: `stimulus_num`, `name_target`, `name_interference`, `utterance`, …

### B2. Synthesize `target_raw`

```bash
python generate_speech_eng.py \
  --input-dir data/reference_eng \
  --voice-info data/generated/voice_info.csv \
  --output-root outputs/stimulus_eng
```

Writes:

`outputs/stimulus_eng/day{i}block{j}/target_raw/day{i}block{j}_stimulus{N}.wav`

Also logs duration outliers (<25 s or >35 s) to:

`data/generated/stimulus_eng_out_of_range.csv`

Resume after interruption:

```bash
python generate_speech_eng.py --start-day 1 --start-block 2 --start-stimulus 5
```

### B3. Stretch to 30 s

```bash
python stretch_audio.py --mode stimulus_eng \
  --root-dir outputs/stimulus_eng \
  --target-sec 30.0
```

### B4. Build final target / interference / mixture

```bash
python stretch_audio.py --mode stimulus_eng_final \
  --root-dir outputs/stimulus_eng \
  --final-output-root outputs/stimulus_eng_final \
  --reference-dir data/reference_eng \
  --precue-dir outputs/precue_eng \
  --precue-sec 4.0
```

**Review** the resulting audio before experiments.

---

## C. Modification / regeneration workflows

### C1. Fix outlier texts (no Typecast credit)

Use measured outlier durations to rewrite utterances toward ~30 s (word-count based prediction; optional second repair pass).

```bash
python regenerate_text.py \
  --outlier-csv data/generated/stimulus_eng_out_of_range.csv \
  --reference-dir data/reference_eng \
  --min-sec 25 --max-sec 35 --target-duration 30
```

Backups of CSVs go to `data/reference_eng/_backup/`.

### C2. Re-TTS only the outlier list

Regenerates WAV for rows listed in the outlier CSV (backs up existing WAV first).

```bash
# preview
python overwrite_modified_speech_eng.py --dry-run

# overwrite
python overwrite_modified_speech_eng.py
```

Log (includes new `duration_sec`):

`data/generated/stimulus_eng_overwrite_log.csv`

### C3. Partial regen via explicit list

`data/regen_list.csv`:

```csv
day,block,stimulus_num
1,3,22
1,3,7
```

```bash
python regenerate_stimulus_eng_from_regen_list.py \
  --regen-list data/regen_list.csv \
  --reference-dir data/reference_eng \
  --output-root outputs/stimulus_eng \
  --overwrite-existing
```

Then re-run stretch + final assembly for affected blocks.

### C4. Out-of-range fix by adjusting TTS tempo

Scan `target_raw`, re-query Typecast with adjusted `audio_tempo` until duration lands in a save window (default 25–30 s).

```bash
python regenerate_stimulus_eng_by_tempo.py \
  --root-dir outputs/stimulus_eng \
  --reference-dir data/reference_eng
```

Then re-run stretch / final if needed.

---

## D. Optional: switched-speaker helpers (legacy / KO-oriented)

| Script | Role |
|--------|------|
| `switch_speaker.py` | Build switched target/interference text CSVs |
| `generate_switched_speech.py` | TTS for switched rows |
| `regenerate_speech.py` | Duration-based re-TTS + text adjust |
| `update_original_csv.py` | Write regenerated utterances back |
| `add_precue_to_switched.py` | Prepend 4 s precue / zero-pad |

Prefer the English day/block flow in section B/C for current experiments.

---

## Utility scripts

| Script | Role |
|--------|------|
| `arrange_text.py` | Sort a text CSV and add a `num` column |
| `filter_stimulus.py` | List stimulus WAVs outside 25–35 s |
| `extract_filtered_text.py` | Pull matching rows from `text_aligned` |
| `extract_same_topic.py` / `modify_topic.py` | Topic clustering / representative remap |
| `plot_hist.py` | Distance histogram helpers (if used locally) |

---

## Suggested QA checklist

1. Text: meaning preserved, quiz still answerable from the utterance  
2. Duration: raw speech roughly 25–35 s before stretch; final pack exactly intended length  
3. Pairing: for stimulus `N`, interference should come from the switched partner (`N±15`)  
4. Levels: listen to mixture for clipping / imbalance  
5. File naming: `day{i}block{j}_stimulus{N}.wav` consistent across target / interference / mixture  

---

## Notes for collaborators

- Script comments and console logs are in English.  
- Model **prompts** that generate Korean content may remain in Korean by design.  
- API keys must never be committed.
