# Sahara challenge demo script

Target duration: 2–3 minutes. Use only built-in judge-mode samples or audio
for which consent and de-identification have been documented.

## 0:00–0:20 — Problem and users

Say: “AfriHealth AI is a prototype for clinician-reviewed documentation of
English-Amharic code-switched consultations. Other language pairs, including
Afaan Oromoo, are future work. It does not replace clinical judgment.”

Show the three care modules and the target frontline workflow.

## 0:20–1:05 — Frontline triage

Open Module 1 and select a built-in sample such as
`GOLD-ETH-001: Pediatric High Fever`. Use “Pre-load Audio Judge Mode” so the
demo is deterministic and does not expose a patient recording.

Show the transcript, extracted medical entities, triage classification, and
referral recommendation. State that the clinician must verify every result
against the source audio and patient context.

## 1:05–1:35 — Clinical intake and follow-up

Open Module 2 to show the editable clinical/EHR intake fields, then Module 3
to show the post-care voice workflow. In Module 2, use a simulated or approved,
de-identified transcript and select “Generate SOAP draft.” This invokes the
agent route when Gemini is configured; otherwise, it displays the manual-review
fallback. Show the returned SOAP draft, ICD-10 candidates, optional TTS
readback, and required clinician sign-off. Email and phone masking is limited
and is not complete de-identification.

## 1:35–2:10 — Benchmark

Open the benchmark matrix and identify its displayed values as a UI fixture,
not independent model runs. The separate 15-case report scores cached
manifest hypotheses for Intron Sahara v2.5, Whisper Tiny, English-only
Wav2Vec2 Base, and Gemini; no new audio inference was run to generate those
reports. Its FAAS is an aggregate composite, not a demographic fairness
evaluation.

Say: “These transcript scores are based on stored hypotheses and are not
population-wide results or fresh audio inference. A final submission should
include the audio, dataset version, model versions, consent status, and
per-sample inference evidence.”

## 2:10–2:35 — Safety and deployment

Show the Responsible AI note. Mention consent, de-identification,
human-in-the-loop review, and server-side API-key storage. Do not display or
type a real API key into the recording.

## Final spoken limitation

“This prototype is a clinical documentation and decision-support aid. It must
not be used as an autonomous diagnostic or prescribing system.”
