# Sahara CodeSwitch Africa submission readiness

## Challenge requirements

The challenge brief requests:

1. a solution description;
2. a short working-prototype demo;
3. code or technical documentation;
4. a comparison across at least three speech models;
5. an ethics/inclusion note; and
6. optional consented, de-identified benchmark audio.

## Current coverage

| Requirement | Project evidence | Status |
| --- | --- | --- |
| Solution description | Product UI and workflow modules in `index.html` | Covered |
| Working prototype | Browser app served by `npm start` | Covered; record a final demo |
| Technical documentation | `README.md`, `main.py`, `server.js`, `benchmark_suite.py` | Covered |
| Primary speech model | Intron Sahara v2.5 | Implemented for live STT and default live benchmark; other providers are optional comparisons |
| Multi-model transcript comparison | `benchmark_suite.py`, `benchmark_report.json`, `evaluation_report_summary.json`, `BENCHMARK_RESULTS.md` | 15-case cached-hypothesis scoring can be rerun from the checked-in manifest; it is not fresh audio inference |
| Agentic clinical workflow | `/api/v1/agent/process`, `src/agents/clinical_agent.py`, SOAP panel in `index.html` | Prototype route and UI are connected; Gemini generates structured drafts when configured, with local ICD-10 candidates and optional Intron TTS. Manual fallback and clinician sign-off remain required |
| Ethics and inclusion | `RESPONSIBLE_AI.md` | Covered; obtain consent evidence |
| Patient-data production controls | Transcript persistence, optional proxy-header check | **Not ready for identifiable patient data: no transcript expiry/deletion or built-in clinician RBAC** |
| Demo video link | External submission artifact | **Still required** |
| Benchmark audio metadata | `BENCHMARK_METADATA_TEMPLATE.csv` | Fill with real consented data |
| Recording and audit handoff | `CLINICAL_RECORDING_PROTOCOL.md`, `TEAM_HANDOFF_CHECKLIST.md` | Ready for team use |

## Benchmark evidence decision

`benchmark_suite.py` scores the cached hypotheses in the 15-case manifest and
generates `benchmark_report.json`, `evaluation_report_summary.json`, and
`BENCHMARK_RESULTS.md` from the same metric calculation. It does not access
audio or run any ASR model. The checked-in inference metadata is marked
`partial`; the OpenAI provider has 0/15 outputs, the selected audio files are
not present in this checkout, and annotator count is blank. The reported
values are transcript-scoring results over stored hypotheses. Cached scoring
can be regenerated from this checkout, but fresh model inference cannot be
independently reproduced without the audio and inference evidence. This is not
a fairness evaluation.

The three baseline columns are Intron Sahara v2.5, Whisper Tiny, and
English-only Wav2Vec2 Base; complete cached Gemini hypotheses are also scored.
Do not describe the Wav2Vec2 model as an Amharic-capable system. Do not claim
the cached hypotheses were regenerated unless the audio, model versions,
decoding settings, and inference logs are available.

The aggregate FAAS formula is not a demographic or subgroup fairness
evaluation. No demographic-group performance breakdown is produced. The
Amharic-English release scope is the only validated product language scope;
Afaan Oromoo remains experimental and is not established as supported.

Before submission, attach or make available the underlying audio and inference
evidence and complete the dataset metadata. Include:

- dataset version and sample count;
- language and code-switch composition;
- audio provenance and consent/de-identification status;
- model versions and decoding settings;
- WER/entity-accuracy definitions;
- per-sample results or an accessible artifact; and
- limitations, including the small current fixture.

Do not claim 0% WER, 100% entity accuracy, or benchmark-wide superiority
unless those figures can be reproduced from the submitted, consented dataset.

## Demo script

Record a 2–3 minute unlisted video showing:

1. the three care modules and intended users;
2. a consented, de-identified sample or the built-in judge-mode sample;
3. partial and final transcription;
4. clinician review of the generated artifacts;
5. the benchmark matrix and methodology;
6. the Responsible AI limitations; and
7. server-side API-key configuration without exposing the key.

## Final pre-submission checklist

- [x] Regenerate aligned 15-case transcript-scoring reports from the manifest.
- [ ] Run fresh, independently reproducible ASR inference on the same consented
      audio for Intron and at least two appropriate comparison models.
- [ ] Record the prepared [DEMO_SCRIPT.md](./DEMO_SCRIPT.md) and add the
      unlisted/public URL to the submission form.
- [ ] Complete annotator count, audio provenance, model versions, decoding
      settings, and accessible per-sample evidence for the 15-case manifest.
- [ ] Follow [CLINICAL_RECORDING_PROTOCOL.md](./CLINICAL_RECORDING_PROTOCOL.md)
      and complete [TEAM_HANDOFF_CHECKLIST.md](./TEAM_HANDOFF_CHECKLIST.md).
- [ ] Verify all model names, versions, and metrics against the actual runs.
- [ ] Deploy with `INTRON_API_KEY` server-side and set `ALLOWED_ORIGINS` to
      the exact production origins.
- [ ] Before any identifiable patient data is used, implement and verify
      transcript retention/deletion controls and clinician authentication with
      role-based access.
- [ ] Have a clinician review generated SOAP, ICD-10, triage, and medication
      outputs before showing them as clinical artifacts.
- [ ] Implement and validate a backend-enforced medication safety gate before
      claiming automated viral/allergy checks or blocked medication suggestions.
- [ ] Remove local secrets and temporary challenge downloads before commit.
