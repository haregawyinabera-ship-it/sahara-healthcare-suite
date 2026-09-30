#!/usr/bin/env python3
"""Multi-dimensional clinical ASR evaluation for code-switched speech.

This script evaluates ASR transcripts using language-aware metrics for
mixed-script clinical conversations. It reports:

- pooled overall WER
- English-only WER
- Amharic/Ge'ez CER
- clinical entity extraction accuracy

WER, English WER, and CER are pooled over their respective reference tokens or
characters. The English-token metric does not assess cross-script transliteration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable

ETHIOPIC_BLOCK = re.compile(r"[\u1200-\u137F]+")
LATIN_WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?")
NON_WORD_RE = re.compile(r"[^\w\u1200-\u137F\s'-]+")


def normalize_text(value: str) -> str:
    value = (value or "").strip()
    return NON_WORD_RE.sub(" ", value).lower().strip()


def tokenize_english(text: str) -> list[str]:
    return LATIN_WORD_RE.findall(normalize_text(text))


def tokenize_ethiopic(text: str) -> str:
    joined = "".join(ch for ch in (text or "") if "\u1200" <= ch <= "\u137F")
    return joined.strip()


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        curr = [i]
        for j, ch_b in enumerate(b, start=1):
            cost = 0 if ch_a == ch_b else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


def word_wer(reference: str, hypothesis: str) -> float:
    ref_tokens = normalize_text(reference).split()
    hyp_tokens = normalize_text(hypothesis).split()
    return word_error_rate(ref_tokens, hyp_tokens)


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    rows = list(range(len(hypothesis) + 1))
    for ref_index, ref_token in enumerate(reference, start=1):
        next_row = [ref_index]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            insertion = next_row[hyp_index - 1] + 1
            deletion = rows[hyp_index] + 1
            substitution = rows[hyp_index - 1] + (ref_token != hyp_token)
            next_row.append(min(insertion, deletion, substitution))
        rows = next_row
    return rows[-1]


def word_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    errors = edit_distance(reference, hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return errors / len(reference)


def char_cer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein_distance(ref, hyp) / max(1, len(ref))


def clinical_entity_recall(reference: str, hypothesis: str) -> float:
    entities = {
        "fever",
        "cough",
        "headache",
        "chest pain",
        "dyspnea",
        "wheezing",
        "amoxicillin",
        "paracetamol",
        "metformin",
        "bp",
        "pulse",
        "temperature",
        "malaria",
        "pneumonia",
        "tb",
        "appendicitis",
        "pregnancy",
    }

    ref_text = normalize_text(reference)
    hyp_text = normalize_text(hypothesis)
    ref_matches = {entity for entity in entities if entity in ref_text}
    if not ref_matches:
        return 1.0

    hits = sum(1 for entity in ref_matches if entity in hyp_text)
    return hits / len(ref_matches)


def calculate_first_partial_latency_ms(first_frame_sent_ms: float | int, first_partial_token_ms: float | int) -> float:
    """Return the elapsed time in milliseconds between the first audio frame and the first partial transcript token."""
    start_ms = float(first_frame_sent_ms)
    end_ms = float(first_partial_token_ms)
    return max(0.0, end_ms - start_ms)


def calculate_partial_token_delay_ms(last_partial_token_ms: float | int, current_partial_token_ms: float | int) -> float:
    return max(0.0, float(current_partial_token_ms) - float(last_partial_token_ms))


def first_partial_latency_sla_met(latency_ms: float | int | None, target_ms: float = 250.0) -> bool:
    if latency_ms is None:
        return False
    return float(latency_ms) <= float(target_ms)


def evaluate_case(reference: str, hypothesis: str, *, first_frame_sent_ms: float | int | None = None, first_partial_token_ms: float | int | None = None, prose_completed_ms: float | int | None = None) -> dict:
    ref_clean = normalize_text(reference)
    hyp_clean = normalize_text(hypothesis)

    english_ref = " ".join(tokenize_english(ref_clean))
    english_hyp = " ".join(tokenize_english(hyp_clean))
    english_wer = word_wer(english_ref, english_hyp)

    ethiopic_ref = tokenize_ethiopic(reference)
    ethiopic_hyp = tokenize_ethiopic(hypothesis)
    ethiopic_cer = char_cer(ethiopic_ref, ethiopic_hyp)

    first_partial_latency_ms = None
    if first_frame_sent_ms is not None and first_partial_token_ms is not None:
        first_partial_latency_ms = calculate_first_partial_latency_ms(first_frame_sent_ms, first_partial_token_ms)

    soap_latency_ms = None
    if first_partial_token_ms is not None and prose_completed_ms is not None:
        soap_latency_ms = max(0.0, float(prose_completed_ms) - float(first_partial_token_ms))

    return {
        "standard_wer": word_wer(reference, hypothesis),
        "english_wer": english_wer,
        "amharic_geez_cer": ethiopic_cer,
        "clinical_entity_recall": clinical_entity_recall(reference, hypothesis),
        "first_partial_latency_ms": first_partial_latency_ms,
        "first_partial_latency_sla_met": first_partial_latency_sla_met(first_partial_latency_ms),
        "soap_generation_latency_ms": soap_latency_ms,
    }


def read_cases(path: Path) -> list[dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
            return payload["cases"]
        raise ValueError("JSON input must be a list of case objects or contain a 'cases' list.")

    if suffix in {".csv"}:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"CSV input {path} is empty.")
        return rows

    raise ValueError(f"Unsupported evaluation input type: {path.suffix}")


def extract_reference_and_hypothesis(row: dict[str, object], ref_column: str, hyp_column: str) -> tuple[str, str]:
    ref = str(row.get(ref_column) or "")
    hyp = str(row.get(hyp_column) or "")
    if not ref and not hyp:
        raise ValueError(f"Row {row} is missing both reference and hypothesis values.")
    if not ref:
        raise ValueError(f"Reference column '{ref_column}' is empty for row: {row}")
    if not hyp:
        raise ValueError(f"Hypothesis column '{hyp_column}' is empty for row: {row}")
    return ref, hyp


def summarize(
    results: Iterable[dict],
    reference_hypothesis_pairs: Iterable[tuple[str, str]],
) -> dict:
    scores = list(results)
    if not scores:
        raise ValueError("No evaluation rows were supplied.")

    pairs = list(reference_hypothesis_pairs)
    if len(pairs) != len(scores):
        raise ValueError("Each evaluation result must have a matching reference/hypothesis pair.")

    def mean(key: str) -> float:
        return sum(item[key] for item in scores) / len(scores)

    overall_wer_errors = 0
    overall_reference_words = 0
    english_wer_errors = 0
    english_reference_words = 0
    geez_cer_errors = 0
    geez_reference_chars = 0
    for reference, hypothesis in pairs:
        ref_words = normalize_text(reference).split()
        hyp_words = normalize_text(hypothesis).split()
        overall_wer_errors += edit_distance(ref_words, hyp_words)
        overall_reference_words += len(ref_words)

        ref_english = tokenize_english(reference)
        hyp_english = tokenize_english(hypothesis)
        english_wer_errors += edit_distance(ref_english, hyp_english)
        english_reference_words += len(ref_english)

        ref_geez = list(tokenize_ethiopic(reference))
        hyp_geez = list(tokenize_ethiopic(hypothesis))
        geez_cer_errors += edit_distance(ref_geez, hyp_geez)
        geez_reference_chars += len(ref_geez)

    def pooled_rate(errors: int, reference_units: int) -> float:
        return errors / reference_units if reference_units else float(errors > 0)

    return {
        "cases_evaluated": len(scores),
        "overall_wer": pooled_rate(overall_wer_errors, overall_reference_words),
        "english_wer": pooled_rate(english_wer_errors, english_reference_words),
        "amharic_geez_cer": pooled_rate(geez_cer_errors, geez_reference_chars),
        "clinical_entity_recall": mean("clinical_entity_recall"),
        "by_case": scores,
    }


def calculate_faas(entity_recall: float, wer: float) -> float:
    """Return the project composite; this is not a demographic fairness metric."""
    return 10.0 * math.log10(max(entity_recall, 0.001) / max(wer, 0.0001))


def evaluate_model_columns(
    rows: list[dict[str, object]],
    reference_column: str,
    model_columns: list[tuple[str, str]],
) -> dict:
    """Score cached transcript columns without opening or requiring audio files."""
    if not rows:
        raise ValueError("No evaluation rows were supplied.")
    if not model_columns:
        raise ValueError("At least one model transcript column is required.")

    models = {}
    for model_name, hypothesis_column in model_columns:
        evaluated = []
        reference_hypothesis_pairs = []
        for index, row in enumerate(rows, start=1):
            reference, hypothesis = extract_reference_and_hypothesis(
                row, reference_column, hypothesis_column
            )
            case = evaluate_case(reference, hypothesis)
            case["case_id"] = str(row.get("case_id") or row.get("sample_id") or f"case_{index}")
            evaluated.append(case)
            reference_hypothesis_pairs.append((reference, hypothesis))

        summary = summarize(evaluated, reference_hypothesis_pairs)
        models[model_name] = {
            "metrics": {
                "overall_wer": summary["overall_wer"],
                "english_wer": summary["english_wer"],
                "amharic_geez_cer": summary["amharic_geez_cer"],
                "clinical_entity_recall": summary["clinical_entity_recall"],
                "faas_score": calculate_faas(
                    summary["clinical_entity_recall"], summary["overall_wer"]
                ),
                "cases_evaluated": summary["cases_evaluated"],
            },
            "by_case": evaluated,
        }

    return {
        "evaluation_type": "language_aware_clinical_asr_comparison",
        "cases_evaluated": len(rows),
        "provenance": {
            "source": "transcript hypotheses supplied in the input file",
            "audio_inference_performed": False,
            "note": "Scores evaluate cached transcript text; model inference was not run.",
        },
        "models_evaluated": models,
    }


def benchmark_report_from_comparison(report: dict) -> dict:
    return {
        name: {
            "overall_wer": round(values["metrics"]["overall_wer"], 4),
            "clinical_entity_recall": round(values["metrics"]["clinical_entity_recall"], 4),
            "faas_score": round(values["metrics"]["faas_score"], 2),
        }
        for name, values in report["models_evaluated"].items()
    }


def benchmark_markdown_from_comparison(report: dict) -> str:
    rows = []
    for name, values in report["models_evaluated"].items():
        metrics = values["metrics"]
        rows.append(
            f"| {name} | {metrics['overall_wer'] * 100:.2f}% | "
            f"{metrics['clinical_entity_recall'] * 100:.2f}% | {metrics['faas_score']:.2f} |"
        )
    return "\n".join([
        "# Clinical ASR Transcript Comparison",
        "",
        f"> **Evidence status:** {report['cases_evaluated']} cached transcript rows were scored; no audio inference was performed.",
        "",
        "| Model | Overall WER ↓ | Clinical Entity Recall ↑ | FAAS Composite |",
        "| :--- | :---: | :---: | :---: |",
        *rows,
        "",
        "FAAS is calculated as 10 * log10(entity recall / WER). It is an aggregate composite, not a demographic fairness measure.",
        "Overall and English WER plus Amharic/Ge'ez CER are pooled over reference words or characters; per-case metrics are also included.",
        "",
    ])


def write_comparison_outputs(
    report: dict,
    *,
    summary_path: str | Path | None = None,
    benchmark_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> None:
    outputs = (
        (summary_path, report),
        (benchmark_path, benchmark_report_from_comparison(report)),
        (markdown_path, benchmark_markdown_from_comparison(report)),
    )
    for path, content in outputs:
        if path is None:
            continue
        output_path = Path(path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            output_path.write_text(content, encoding="utf-8")
        else:
            output_path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate code-switched clinical ASR transcripts using language-aware "
            "WER/CER/MER metrics instead of a single standard WER score."
        )
    )
    parser.add_argument("--input", required=True, help="CSV/JSON file with reference and hypothesis columns.")
    parser.add_argument("--reference-column", default="reference_transcript", help="Reference transcript column name.")
    parser.add_argument("--hypothesis-column", default="hypothesis_transcript", help="Hypothesis transcript column name.")
    parser.add_argument("--first-frame-column", default="first_frame_sent_ms", help="Optional column containing the first audio-frame send time in ms.")
    parser.add_argument("--first-partial-column", default="first_partial_token_ms", help="Optional column containing the first partial-token receive time in ms.")
    parser.add_argument("--soap-complete-column", default="soap_complete_ms", help="Optional column containing SOAP generation completion time in ms.")
    parser.add_argument("--output", help="Optional JSON report output path.")
    parser.add_argument(
        "--model-column",
        action="append",
        default=[],
        metavar="MODEL=COLUMN",
        help="Compare one cached hypothesis column; may be repeated.",
    )
    parser.add_argument("--benchmark-report", help="Optional legacy-format comparison JSON output.")
    parser.add_argument("--markdown-report", help="Optional comparison table output.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Evaluation input not found: {input_path}")

    rows = read_cases(input_path)
    if args.model_column:
        model_columns = []
        for item in args.model_column:
            if "=" not in item:
                parser.error(f"Invalid --model-column {item!r}; expected MODEL=COLUMN")
            model_name, column = item.split("=", 1)
            if not model_name.strip() or not column.strip():
                parser.error(f"Invalid --model-column {item!r}; expected non-empty MODEL=COLUMN")
            model_columns.append((model_name.strip(), column.strip()))

        report = evaluate_model_columns(rows, args.reference_column, model_columns)
        report["dataset"] = input_path.name
        print(json.dumps({
            name: values["metrics"] for name, values in report["models_evaluated"].items()
        }, ensure_ascii=False, indent=2))
        write_comparison_outputs(
            report,
            summary_path=args.output,
            benchmark_path=args.benchmark_report,
            markdown_path=args.markdown_report,
        )
        return

    evaluated = []
    reference_hypothesis_pairs = []
    for index, row in enumerate(rows, start=1):
        ref, hyp = extract_reference_and_hypothesis(row, args.reference_column, args.hypothesis_column)
        first_frame_ms = row.get(args.first_frame_column)
        first_partial_ms = row.get(args.first_partial_column)
        soap_complete_ms = row.get(args.soap_complete_column)
        result = evaluate_case(
            ref,
            hyp,
            first_frame_sent_ms=float(first_frame_ms) if first_frame_ms not in (None, "") else None,
            first_partial_token_ms=float(first_partial_ms) if first_partial_ms not in (None, "") else None,
            prose_completed_ms=float(soap_complete_ms) if soap_complete_ms not in (None, "") else None,
        )
        result["case_id"] = str(row.get("case_id") or f"case_{index}")
        evaluated.append(result)
        reference_hypothesis_pairs.append((ref, hyp))

    summary = summarize(evaluated, reference_hypothesis_pairs)
    latency_values = [case.get("first_partial_latency_ms") for case in evaluated if case.get("first_partial_latency_ms") is not None]
    mean_first_partial_latency = sum(latency_values) / len(latency_values) if latency_values else None
    latency_sla_pass_rate = None
    if latency_values:
        latency_sla_pass_rate = sum(1 for value in latency_values if first_partial_latency_sla_met(value)) / len(latency_values)
    report = {
        "dataset": input_path.name,
        "evaluation_type": "language_aware_clinical_asr",
        "metrics": {
            "overall_wer": round(summary["overall_wer"], 4),
            "english_wer": round(summary["english_wer"], 4),
            "amharic_geez_cer": round(summary["amharic_geez_cer"], 4),
            "clinical_entity_recall": round(summary["clinical_entity_recall"], 4),
            "mean_first_partial_latency_ms": round(mean_first_partial_latency, 2) if mean_first_partial_latency is not None else None,
            "first_partial_latency_sla_target_ms": 250,
            "first_partial_latency_sla_pass_rate": round(latency_sla_pass_rate, 4) if latency_sla_pass_rate is not None else None,
        },
        "cases_evaluated": summary["cases_evaluated"],
        "by_case": evaluated,
    }

    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nReport written to {output_path}")


if __name__ == "__main__":
    main()
