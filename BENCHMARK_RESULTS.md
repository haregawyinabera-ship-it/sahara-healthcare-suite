# Clinical ASR Transcript Comparison

> **Evidence status:** 15 cached transcript rows were scored; no audio inference was performed.

| Model | Overall WER ↓ | Clinical Entity Recall ↑ | FAAS Composite |
| :--- | :---: | :---: | :---: |
| Intron Sahara v2.5 | 41.75% | 80.00% | 2.82 |
| OpenAI Whisper Tiny | 95.63% | 60.00% | -2.02 |
| Meta Wav2Vec2 Base 960h (English) | 113.59% | 46.67% | -3.86 |
| Google Gemini gemini-flash-latest | 21.36% | 100.00% | 6.70 |

FAAS is calculated as 10 * log10(entity recall / WER). It is an aggregate composite, not a demographic fairness measure.
Overall and English WER plus Amharic/Ge'ez CER are pooled over reference words or characters; per-case metrics are also included.
