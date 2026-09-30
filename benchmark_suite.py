import os
import json
import math
import asyncio
import argparse
import csv
from pathlib import Path
from statistics import fmean
from run_full_evaluation import (
    benchmark_report_from_comparison,
    evaluate_model_columns,
    write_comparison_outputs,
)

try:
    import jiwer
except ImportError:
    jiwer = None

DATASET_INDEX_PATH = os.getenv("DATASET_INDEX", "./evaluation_dataset.json")
OUTPUT_REPORT_PATH = os.getenv("OUTPUT_REPORT", "./benchmark_report.json")
OUTPUT_MARKDOWN_PATH = os.getenv("OUTPUT_MARKDOWN", "./BENCHMARK_RESULTS.md")
AFRISWITCH_ROOT = Path(os.getenv("AFRISWITCH_ROOT", "./clinical_validation/afriswitch"))

INTRON_API_KEY = os.getenv("INTRON_API_KEY", "")

CLINICAL_ENTITIES = [
    "fever", "cough", "headache", "hypertension", "diabetes", "paracetamol",
    "amoxicillin", "metformin", "bp", "pulse", "chills", "tb", "malaria",
    "pneumonia", "dyspnea", "tachycardia", "tuberculosis", "mg", "ml"
]

def calculate_wer(reference: str, hypothesis: str) -> float:
    ref_clean = reference.lower().strip()
    hyp_clean = hypothesis.lower().strip()
    if not ref_clean:
        return 0.0 if not hyp_clean else 1.0
    if jiwer:
        return float(jiwer.wer(ref_clean, hyp_clean))

    ref_words = ref_clean.split()
    hyp_words = hyp_clean.split()
    distances = list(range(len(hyp_words) + 1))
    for ref_word in ref_words:
        next_distances = [distances[0] + 1]
        for index, hyp_word in enumerate(hyp_words, start=1):
            substitution = distances[index - 1] + (ref_word != hyp_word)
            insertion = next_distances[index - 1] + 1
            deletion = distances[index] + 1
            next_distances.append(min(substitution, insertion, deletion))
        distances = next_distances
    return distances[-1] / len(ref_words)

def calculate_entity_recall(reference: str, hypothesis: str) -> float:
    ref_words = set(reference.lower().split())
    hyp_words = set(hypothesis.lower().split())
    target_entities = [e for e in CLINICAL_ENTITIES if e in ref_words or any(e in w for w in ref_words)]
    if not target_entities:
        return 1.0
    matches = sum(1 for entity in target_entities if any(entity in w for w in hyp_words))
    return matches / len(target_entities)

def calculate_faas(entity_recall: float, wer: float) -> float:
    if wer <= 0.0001:
        wer = 0.0001
    if entity_recall <= 0.0:
        entity_recall = 0.001
    return float(10.0 * math.log10(entity_recall / wer))

DEFAULT_BENCHMARK_SAMPLES = [
    {
        "id": "sample_001",
        "audio_path": "./samples/sample_001.wav",
        "reference": "patient unique identification. patient presents with severe headache and fever spanning 3 days. prescribed paracetamol 500mg twice daily.",
        "language_pair": "English-Amharic Code-Switch",
        "hypotheses": {
            "Intron Sahara v2.5": "patient unique identification. patient presents with severe headache and fever spanning 3 days. prescribed paracetamol 500mg twice daily.",
            "OpenAI Whisper Tiny": "patient unique identification patient present with severe headache and high fever 3 days prescribed paracetamol 500 daily",
            "Meta Wav2Vec2 Base 960h (English)": "patient unique identification patient severe headache fever 3 days prescribed paracetamol"
        }
    },
    {
        "id": "sample_002",
        "audio_path": "./samples/sample_002.wav",
        "reference": "የጤና ተቋም። chief complaint is chest pain with short breath. clinical assessment shows blood pressure 140 over 90.",
        "language_pair": "English-Amharic Code-Switch",
        "hypotheses": {
            "Intron Sahara v2.5": "የጤና ተቋም። chief complaint is chest pain with short breath. clinical assessment shows blood pressure 140 over 90.",
            "OpenAI Whisper Tiny": "የጤና ተቋም chief complaint chest pain short breath blood pressure 140 over 90",
            "Meta Wav2Vec2 Base 960h (English)": "chief complaint chest pain short breath blood pressure 140 90"
        }
    },
    {
        "id": "sample_003",
        "audio_path": "./samples/sample_003.wav",
        "reference": "patient has suspected malaria and pneumonia. recommended amoxicillin 500mg and urgent lab workup.",
        "language_pair": "English Clinical Standard",
        "hypotheses": {
            "Intron Sahara v2.5": "patient has suspected malaria and pneumonia. recommended amoxicillin 500mg and urgent lab workup.",
            "OpenAI Whisper Tiny": "patient suspected malaria and pneumonia recommended amoxicillin 500mg urgent lab workup",
            "Meta Wav2Vec2 Base 960h (English)": "patient suspect malaria pneumonia recommended amoxicillin lab workup"
        }
    },
    {
        "id": "sample_004",
        "audio_path": "./samples/sample_004.wav",
        "reference": "patient reports fever and cough for 2 days. oxygen saturation is low and there is wheezing in the lungs.",
        "language_pair": "English Clinical Standard",
        "hypotheses": {
            "Intron Sahara v2.5": "patient reports fever and cough for 2 days. oxygen saturation is low and there is wheezing in the lungs.",
            "OpenAI Whisper Tiny": "patient reports fever cough for 2 days oxygen saturation low wheezing in lungs",
            "Meta Wav2Vec2 Base 960h (English)": "patient fever cough 2 days low oxygen saturation wheezing lungs"
        }
    },
    {
        "id": "sample_005",
        "audio_path": "./samples/sample_005.wav",
        "reference": "የህሙም በሽታ አብዛኛውን ጊዜ በእግር ህመም እና በቫይታሚን እጥረት ተገኝቷል። doctor advised metformin 500mg once daily and hydration.",
        "language_pair": "English-Amharic Code-Switch",
        "hypotheses": {
            "Intron Sahara v2.5": "የህሙም በሽታ አብዛኛውን ጊዜ በእግር ህመም እና በቫይታሚን እጥረት ተገኝቷል። doctor advised metformin 500mg once daily and hydration.",
            "OpenAI Whisper Tiny": "ህመም በእግር ህመም እና ቫይታሚን እጥረት ተገኝቷል doctor advised metformin 500 once daily hydration",
            "Meta Wav2Vec2 Base 960h (English)": "ህመም በእግር ህመም እና ቫይታሚን እጥረት doctor advised metformin once daily"
        }
    },
    {
        "id": "sample_006",
        "audio_path": "./samples/sample_006.wav",
        "reference": "patient reports abdominal pain and diarrhea for 4 days. blood pressure is 110 over 70 and pulse is elevated.",
        "language_pair": "English Clinical Standard",
        "hypotheses": {
            "Intron Sahara v2.5": "patient reports abdominal pain and diarrhea for 4 days. blood pressure is 110 over 70 and pulse is elevated.",
            "OpenAI Whisper Tiny": "patient reports abdominal pain diarrhea 4 days blood pressure 110 over 70 pulse elevated",
            "Meta Wav2Vec2 Base 960h (English)": "patient abdominal pain diarrhea 4 days blood pressure 110 70 pulse elevated"
        }
    }
]


def load_benchmark_samples() -> list[dict]:
    dataset_dir = Path(os.getenv("BENCHMARK_DATASET_DIR", "./benchmark_data"))
    manifest_path = dataset_dir / "manifest.csv"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
            rows = list(csv.DictReader(manifest_file))
        inference_metadata_path = dataset_dir / "inference_metadata.json"
        inference_models = {}
        if inference_metadata_path.is_file():
            inference_models = json.loads(inference_metadata_path.read_text(encoding="utf-8")).get("models", {})
        openai_model_id = inference_models.get("openai", {}).get("model_id", "gpt-4o-mini-transcribe")
        gemini_model_id = inference_models.get("gemini", {}).get("model_id", "gemini-flash-latest")
        items = []
        model_columns = {
            "Intron Sahara v2.5": "intron_hypothesis",
            "OpenAI Whisper Tiny": "whisper_hypothesis",
            "Meta Wav2Vec2 Base 960h (English)": "wav2vec2_hypothesis",
            f"OpenAI {openai_model_id}": "openai_hypothesis",
            f"Google Gemini {gemini_model_id}": "gemini_hypothesis",
        }
        for row in rows:
            sample_id = (row.get("sample_id") or row.get("id") or "").strip()
            audio_file = (row.get("audio_file") or row.get("audio_path") or "").strip()
            if not sample_id or not audio_file:
                continue
            audio_path = (dataset_dir / audio_file).as_posix() if not audio_file.startswith("/") else audio_file
            items.append({
                "id": sample_id,
                "case_id": (row.get("case_id") or sample_id).strip(),
                "audio_path": audio_path,
                "source_audio_file": (row.get("source_audio_file") or "").strip(),
                "reference": (row.get("reference") or row.get("gold_standard") or "").strip(),
                "target_terms": (row.get("target_terms") or "").strip(),
                "language_pair": (row.get("language_pair") or "Not provided").strip(),
                "reference_source": (row.get("reference_source") or "").strip(),
                "consent_obtained": (row.get("consent_obtained") or "").strip(),
                "de_identified": (row.get("de_identified") or "").strip(),
                "annotator_count": (row.get("annotator_count") or "").strip(),
                "reference_verified": (row.get("reference_verified") or "").strip().lower() in {"true", "1", "yes"},
                "verified_gold_standard": (row.get("verified_gold_standard") or "").strip().lower() in {"true", "1", "yes"},
                "duplicate_of": (row.get("duplicate_of") or "").strip(),
                "hypotheses": {
                    model: row[column].strip()
                    for model, column in model_columns.items()
                    if row.get(column, "").strip()
                },
            })
        return items

    json_manifest = dataset_dir / "manifest.json"
    if json_manifest.exists():
        payload = json.loads(json_manifest.read_text(encoding="utf-8"))
        if isinstance(payload, list) and payload:
            return payload

    return DEFAULT_BENCHMARK_SAMPLES


BENCHMARK_SAMPLES = load_benchmark_samples()


def unique_benchmark_samples(samples: list[dict]) -> list[dict]:
    return [sample for sample in samples if not sample.get("duplicate_of")]


def available_optional_models(samples: list[dict]) -> list[str]:
    base_models = {
        "Intron Sahara v2.5",
        "OpenAI Whisper Tiny",
        "Meta Wav2Vec2 Base 960h (English)",
    }
    optional_models = []
    for sample in samples:
        for model in sample.get("hypotheses", {}):
            if (
                model not in base_models
                and model.startswith(("OpenAI ", "Google Gemini "))
                and model not in optional_models
            ):
                optional_models.append(model)
    available = []
    for model in optional_models:
        counts = [bool(sample.get("hypotheses", {}).get(model)) for sample in samples]
        if any(counts) and not all(counts):
            raise ValueError(f"Partial hypotheses exist for {model}; complete all samples or clear that provider column.")
        if counts and all(counts):
            available.append(model)
    return available


def validate_benchmark_samples(samples: list[dict], models: list[str]) -> None:
    if not samples:
        raise ValueError("No benchmark samples are available.")
    incomplete = [
        item["id"]
        for item in samples
        if not item.get("reference")
        or (
            "reference_verified" in item
            and not item["reference_verified"]
            and not item.get("verified_gold_standard")
        )
        or any(not item.get("hypotheses", {}).get(model) for model in models)
    ]
    if incomplete:
        sample_ids = ", ".join(incomplete[:5])
        suffix = "..." if len(incomplete) > 5 else ""
        raise ValueError(
            f"Cannot calculate benchmark metrics: {len(incomplete)} sample(s) lack "
            f"a verified reference transcript or model hypothesis ({sample_ids}{suffix})."
        )

def run_afriswitch_pilot(root: Path) -> None:
    """Validate the imported pilot and report reference coverage only."""
    manifest_dir = root / "manifests"
    metadata_path = root / "import_metadata.json"
    if not manifest_dir.is_dir() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"AfriSwitch import not found at {root}. Run afriswitch_import.py first."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    configs = {}
    total_duration = 0.0
    for manifest_path in sorted(manifest_dir.glob("*.csv")):
        rows = list(csv.DictReader(manifest_path.open(encoding="utf-8", newline="")))
        missing_audio = sum(
            not (root / row["local_audio"]).is_file()
            for row in rows
        )
        duration = sum(float(row["duration"]) for row in rows if row.get("duration"))
        switches = sum(int(row["num_switch_points"]) for row in rows if row.get("num_switch_points"))
        configs[manifest_path.stem] = {
            "utterances": len(rows),
            "audio_files": len(rows) - missing_audio,
            "missing_audio": missing_audio,
            "duration_seconds": round(duration, 2),
            "switch_points": switches,
        }
        total_duration += duration

    report = {
        "dataset_id": metadata["dataset_id"],
        "dataset_revision": metadata["dataset_revision"],
        "split": metadata["split"],
        "license": metadata["license"],
        "configs": configs,
        "total_utterances": sum(item["utterances"] for item in configs.values()),
        "total_duration_seconds": round(total_duration, 2),
        "evaluation_status": "reference_only",
        "note": "No model hypotheses were supplied; WER and model rankings are intentionally not reported.",
    }
    output_path = root / "afriswitch_pilot_report.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Pilot report exported to {output_path}")

async def run_benchmark():
    models = ["Intron Sahara v2.5", "OpenAI Whisper Tiny", "Meta Wav2Vec2 Base 960h (English)"]
    scoring_samples = unique_benchmark_samples(BENCHMARK_SAMPLES)
    models.extend(available_optional_models(scoring_samples))
    validate_benchmark_samples(scoring_samples, models)

    manifest_path = Path(os.getenv("BENCHMARK_DATASET_DIR", "./benchmark_data")) / "manifest.csv"
    if manifest_path.is_file():
        rows = [
            {
                "case_id": sample["case_id"],
                "reference": sample["reference"],
                **sample["hypotheses"],
            }
            for sample in scoring_samples
        ]
        report = evaluate_model_columns(
            rows,
            "reference",
            [(model, model) for model in models],
        )
        report["dataset"] = manifest_path.name
        metadata_path = manifest_path.parent / "inference_metadata.json"
        if metadata_path.is_file():
            inference_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            report["inference_run_status"] = inference_metadata.get("run_status")
            report["inference_generated_this_run"] = inference_metadata.get("generated_this_run", {})

        write_comparison_outputs(
            report,
            summary_path=os.getenv("OUTPUT_EVALUATION_SUMMARY", "./evaluation_report_summary.json"),
            benchmark_path=OUTPUT_REPORT_PATH,
            markdown_path=OUTPUT_MARKDOWN_PATH,
        )
        print(json.dumps(benchmark_report_from_comparison(report), ensure_ascii=False, indent=2))
        print(
            "Scored cached manifest hypotheses only; audio inference was not run. "
            f"Reports exported to {OUTPUT_REPORT_PATH}, {OUTPUT_MARKDOWN_PATH}, "
            f"and {os.getenv('OUTPUT_EVALUATION_SUMMARY', './evaluation_report_summary.json')}."
        )
        return

    print("============================================================")
    print("Starting Multi-Model Speech Recognition Benchmark...")
    print("Models: Intron Sahara v2.5 | OpenAI Whisper Tiny | Meta Wav2Vec2 Base 960h (English)")
    print("============================================================")

    results = {m: {"wers": [], "entity_recalls": []} for m in models}

    for item in scoring_samples:
        ref = item["reference"]
        for model_name in models:
            hyp = item["hypotheses"][model_name]
            wer = calculate_wer(ref, hyp)
            entity_recall = calculate_entity_recall(ref, hyp)
            results[model_name]["wers"].append(wer)
            results[model_name]["entity_recalls"].append(entity_recall)

    summary = {}
    print("\n============================================================")
    print("FINAL BENCHMARK RESULTS")
    print("============================================================")

    for m in models:
        mean_wer = fmean(results[m]["wers"])
        mean_entity_recall = fmean(results[m]["entity_recalls"])
        faas = calculate_faas(entity_recall=mean_entity_recall, wer=mean_wer)
        
        summary[m] = {
            "mean_wer": round(mean_wer, 4),
            "clinical_entity_recall": round(mean_entity_recall, 4),
            "faas_score": round(faas, 2)
        }
        print(f"Model: {m}")
        print(f"  - Mean WER: {summary[m]['mean_wer'] * 100:.2f}%")
        print(f"  - Clinical Entity Recall: {summary[m]['clinical_entity_recall'] * 100:.2f}%")
        print(f"  - FAAS Score: {summary[m]['faas_score']} dB\n")

    with open(OUTPUT_REPORT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    manifest_path = Path(os.getenv("BENCHMARK_DATASET_DIR", "./benchmark_data")) / "manifest.csv"
    if manifest_path.is_file():
        report_title = "# Clinical Audio Dataset Benchmark Report"
        evidence_note = (
            f"Results were scored from {len(scoring_samples)} manifest-selected recordings against verified references. "
            "Model checkpoints and inference settings are recorded in `benchmark_data/inference_metadata.json`."
        )
        intron_status = "Measured hosted ASR"
        whisper_status = "Measured local ASR"
        wav2vec_status = "English-only local baseline; checkpoint in inference metadata"
    else:
        report_title = "# Fixture Speech Recognition Benchmark Report"
        evidence_note = (
            "This is a reproducible software fixture, not an independent audio benchmark. "
            "Hypotheses are embedded in `benchmark_suite.py`; do not present these values as production performance."
        )
        intron_status = "Fixture reference"
        whisper_status = "Fixture baseline"
        wav2vec_status = "Fixture baseline"

    model_status = {
        "Intron Sahara v2.5": intron_status,
        "OpenAI Whisper Tiny": whisper_status,
        "Meta Wav2Vec2 Base 960h (English)": wav2vec_status,
    }
    report_rows = []
    for model in models:
        values = summary[model]
        report_rows.append(
            f"| {model} | {values['mean_wer'] * 100:.2f}% | "
            f"{values['clinical_entity_recall'] * 100:.2f}% | {values['faas_score']:.2f} | "
            f"{model_status.get(model, 'Measured ASR')} |"
        )
    model_table_rows = "\n".join(report_rows)

    md_content = f"""{report_title}

> **Evidence status:** {evidence_note}

| Model | Average WER ↓ | Clinical Entity Recall ↑ | FAAS Score (dB) ↑ | Status |
| :--- | :---: | :---: | :---: | :---: |
{model_table_rows}

### Evaluation Methodology
1. **Word Error Rate (WER)**: Normalized string distance metric (S + D + I) / N.
2. **Clinical Entity Recall**: Recall rate of reference clinical terms (symptoms, dosages, diagnoses).
3. **FAAS composite**: Calculated as 10 * log10(Clinical Entity Recall / WER); this aggregate score is not a demographic fairness metric.

The separate 15-case clinical validation baseline reported 56.38% mean WER,
44.33% target-term recall, and critical-term misses in 6 cases. That result is
for clinician review only and does not support autonomous clinical use.
"""
    with open(OUTPUT_MARKDOWN_PATH, "w") as f:
        f.write(md_content)

    print(f"Report exported to {OUTPUT_REPORT_PATH} and {OUTPUT_MARKDOWN_PATH}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the fixture benchmark or validate the AfriSwitch pilot.")
    parser.add_argument(
        "--afriswitch-pilot",
        action="store_true",
        help="Generate a reference-only report from the imported AfriSwitch pilot.",
    )
    parser.add_argument(
        "--afriswitch-root",
        default=str(AFRISWITCH_ROOT),
        help="Path to the imported AfriSwitch directory.",
    )
    args = parser.parse_args()
    if args.afriswitch_pilot:
        run_afriswitch_pilot(Path(args.afriswitch_root))
    else:
        asyncio.run(run_benchmark())
