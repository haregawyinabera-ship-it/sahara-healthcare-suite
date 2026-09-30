import asyncio
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

import benchmark_suite
import generate_benchmark_hypotheses
import run_full_evaluation


class BenchmarkMetricTests(unittest.TestCase):
    def test_entity_metrics_are_reference_recall(self):
        self.assertEqual(benchmark_suite.calculate_entity_recall("fever cough", "fever"), 0.5)
        self.assertEqual(run_full_evaluation.clinical_entity_recall("fever cough", "fever"), 0.5)

    def test_multi_model_evaluator_scores_transcripts_without_audio(self):
        rows = [{
            "sample_id": "CS-01",
            "reference": "fever cough",
            "intron_hypothesis": "fever cough",
            "whisper_hypothesis": "fever",
        }]

        report = run_full_evaluation.evaluate_model_columns(
            rows,
            "reference",
            [
                ("Intron", "intron_hypothesis"),
                ("Whisper", "whisper_hypothesis"),
            ],
        )

        self.assertEqual(report["cases_evaluated"], 1)
        self.assertFalse(report["provenance"]["audio_inference_performed"])
        self.assertEqual(report["models_evaluated"]["Intron"]["metrics"]["overall_wer"], 0.0)
        self.assertEqual(report["models_evaluated"]["Whisper"]["metrics"]["overall_wer"], 0.5)
        self.assertNotIn("transliteration_mer", report["models_evaluated"]["Whisper"]["metrics"])
        self.assertNotIn("transliteration_mer", report["models_evaluated"]["Whisper"]["by_case"][0])

    def test_overall_wer_is_pooled_by_reference_word_count(self):
        rows = [
            {"reference": "one two", "hypothesis": "one"},
            {
                "reference": "a b c d e f g h i j",
                "hypothesis": "a b c d e f g h i j",
            },
        ]

        report = run_full_evaluation.evaluate_model_columns(
            rows,
            "reference",
            [("Test", "hypothesis")],
        )

        metrics = report["models_evaluated"]["Test"]["metrics"]
        self.assertAlmostEqual(metrics["overall_wer"], 1 / 12)
        self.assertAlmostEqual(metrics["english_wer"], 1 / 12)

    def test_comparison_outputs_share_one_metric_source(self):
        rows = [{
            "case_id": "CS-01",
            "reference": "fever cough",
            "hypothesis": "fever",
        }]
        report = run_full_evaluation.evaluate_model_columns(
            rows,
            "reference",
            [("Whisper", "hypothesis")],
        )

        with tempfile.TemporaryDirectory() as output_dir:
            summary_path = Path(output_dir) / "summary.json"
            benchmark_path = Path(output_dir) / "benchmark.json"
            markdown_path = Path(output_dir) / "results.md"
            run_full_evaluation.write_comparison_outputs(
                report,
                summary_path=summary_path,
                benchmark_path=benchmark_path,
                markdown_path=markdown_path,
            )
            summary = __import__("json").loads(summary_path.read_text(encoding="utf-8"))
            benchmark = __import__("json").loads(benchmark_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        metric = summary["models_evaluated"]["Whisper"]["metrics"]
        self.assertEqual(benchmark["Whisper"]["overall_wer"], round(metric["overall_wer"], 4))
        self.assertEqual(benchmark["Whisper"]["faas_score"], round(metric["faas_score"], 2))
        self.assertIn("no audio inference was performed", markdown)

    def test_active_benchmark_dataset_contains_more_than_four_samples(self):
        self.assertGreaterEqual(len(benchmark_suite.BENCHMARK_SAMPLES), 6)

    def test_dataset_manifest_uses_one_recording_per_audited_case(self):
        samples = benchmark_suite.BENCHMARK_SAMPLES
        expected_cases = {f"CS-{number:02d}" for number in range(1, 16)}

        self.assertEqual(len(samples), 15)
        self.assertEqual({sample["case_id"] for sample in samples}, expected_cases)
        self.assertFalse(any("'" in Path(sample["audio_path"]).name for sample in samples))
        self.assertFalse(any(sample.get("duplicate_of") for sample in samples))

    def test_manifest_does_not_copy_references_to_model_hypotheses(self):
        with tempfile.TemporaryDirectory() as dataset_dir:
            manifest = Path(dataset_dir) / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "sample_id",
                        "case_id",
                        "audio_file",
                        "source_audio_file",
                        "reference",
                        "target_terms",
                        "reference_source",
                        "reference_verified",
                        "verified_gold_standard",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "sample_id": "audio_only",
                    "case_id": "CS-01",
                    "audio_file": "samples/audio.m4a",
                    "source_audio_file": "samples/audio.m4a",
                    "reference": "verified transcript",
                    "target_terms": "fever",
                    "reference_source": "reference.csv",
                    "reference_verified": "false",
                    "verified_gold_standard": "false",
                })
            with patch.dict("os.environ", {"BENCHMARK_DATASET_DIR": dataset_dir}):
                samples = benchmark_suite.load_benchmark_samples()

        self.assertEqual(samples[0]["reference"], "verified transcript")
        self.assertEqual(samples[0]["case_id"], "CS-01")
        self.assertEqual(samples[0]["source_audio_file"], "samples/audio.m4a")
        self.assertEqual(samples[0]["target_terms"], "fever")
        self.assertEqual(samples[0]["reference_source"], "reference.csv")
        self.assertFalse(samples[0]["reference_verified"])
        self.assertEqual(samples[0]["hypotheses"], {})

    def test_hypothesis_runner_adds_optional_provider_columns(self):
        with tempfile.TemporaryDirectory() as dataset_dir:
            manifest = Path(dataset_dir) / "manifest.csv"
            manifest.write_text("sample_id,audio_file\nCS-01,audio.wav\n", encoding="utf-8")

            fieldnames, rows = generate_benchmark_hypotheses.load_manifest(manifest)

        self.assertIn("openai_hypothesis", fieldnames)
        self.assertIn("gemini_hypothesis", fieldnames)
        self.assertEqual(rows[0]["openai_hypothesis"], "")
        self.assertEqual(rows[0]["gemini_hypothesis"], "")

    def test_benchmark_refuses_incomplete_audio_dataset(self):
        with self.assertRaisesRegex(ValueError, "lack a verified reference transcript or model hypothesis"):
            benchmark_suite.validate_benchmark_samples(
                [{
                    "id": "audio_only",
                    "reference": "",
                    "reference_verified": False,
                    "hypotheses": {},
                }],
                ["Intron Sahara v2.5"],
            )

    def test_benchmark_refuses_unverified_reference_transcripts(self):
        models = ["Intron Sahara v2.5"]
        sample = {
            "id": "unverified",
            "reference": "fever",
            "reference_verified": False,
            "hypotheses": {models[0]: "fever"},
        }

        with self.assertRaisesRegex(ValueError, "lack a verified reference transcript"):
            benchmark_suite.validate_benchmark_samples([sample], models)

    def test_hypothesis_runner_requires_consent_and_reference_approval(self):
        sample = {
            "sample_id": "CS-01",
            "consent_obtained": "true",
            "de_identified": "",
            "reference_verified": "true",
            "verified_gold_standard": "true",
        }

        with self.assertRaisesRegex(ValueError, "Required consent/de-identification/reference approvals"):
            generate_benchmark_hypotheses.require_authorized_dataset([sample], include_intron=True)

    def test_duplicate_recording_is_excluded_from_scoring(self):
        samples = [
            {"id": "original", "duplicate_of": ""},
            {"id": "copy", "duplicate_of": "original"},
        ]

        self.assertEqual(
            [sample["id"] for sample in benchmark_suite.unique_benchmark_samples(samples)],
            ["original"],
        )

    def test_optional_provider_scores_require_complete_sample_coverage(self):
        complete = [
            {"hypotheses": {"OpenAI gpt-4o-mini-transcribe": "text"}},
            {"hypotheses": {"OpenAI gpt-4o-mini-transcribe": "text"}},
        ]
        self.assertEqual(
            benchmark_suite.available_optional_models(complete),
            ["OpenAI gpt-4o-mini-transcribe"],
        )

        partial = [
            {"hypotheses": {"OpenAI gpt-4o-mini-transcribe": "text"}},
            {"hypotheses": {}},
        ]
        with self.assertRaisesRegex(ValueError, "Partial hypotheses exist"):
            benchmark_suite.available_optional_models(partial)

    def test_hosted_provider_retries_transient_503(self):
        attempts = {"count": 0}

        async def provider():
            attempts["count"] += 1
            if attempts["count"] < 3:
                request = httpx.Request("POST", "https://provider.invalid")
                response = httpx.Response(503, request=request, headers={"Retry-After": "1"})
                raise httpx.HTTPStatusError("temporarily unavailable", request=request, response=response)
            return "transcribed audio"

        async def run_provider():
            with patch("generate_benchmark_hypotheses.asyncio.sleep", new=AsyncMock()) as sleep:
                result = await generate_benchmark_hypotheses.call_with_transient_retries(provider)
                self.assertEqual(sleep.await_count, 2)
                return result

        self.assertEqual(asyncio.run(run_provider()), "transcribed audio")
        self.assertEqual(attempts["count"], 3)

    def test_gemini_runner_spaces_requests_and_resumes_populated_rows(self):
        async def run_provider():
            with tempfile.TemporaryDirectory() as dataset_dir:
                root = Path(dataset_dir)
                audio = root / "sample.wav"
                audio.write_bytes(b"audio")
                manifest = root / "manifest.csv"
                fieldnames = ["sample_id", "audio_file", "gemini_hypothesis"]
                rows = [
                    {"sample_id": "saved", "audio_file": "sample.wav", "gemini_hypothesis": "already saved"},
                    {"sample_id": "pending", "audio_file": "sample.wav", "gemini_hypothesis": ""},
                ]
                manifest.write_text("", encoding="utf-8")
                import main
                with patch.object(main, "_benchmark_gemini", new=AsyncMock(return_value="new transcript")) as provider:
                    with patch("generate_benchmark_hypotheses.asyncio.sleep", new=AsyncMock()) as sleep:
                        completed, errors = await generate_benchmark_hypotheses.run_hosted_provider(
                            "gemini", manifest, root, fieldnames, rows, False, request_interval_seconds=15
                        )
                self.assertEqual(provider.await_count, 1)
                self.assertEqual(sleep.await_args.args, (15,))
                return completed, errors

        completed, errors = asyncio.run(run_provider())
        self.assertEqual(completed, 1)
        self.assertEqual(errors, [])

    def test_first_partial_metric_is_latency_not_faas(self):
        result = run_full_evaluation.evaluate_case(
            "fever cough",
            "fever",
            first_frame_sent_ms=100,
            first_partial_token_ms=350,
        )

        self.assertEqual(result["first_partial_latency_ms"], 250)
        self.assertTrue(result["first_partial_latency_sla_met"])
        self.assertNotIn("faas_ms", result)


if __name__ == "__main__":
    unittest.main()