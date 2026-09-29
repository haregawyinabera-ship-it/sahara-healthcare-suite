import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

import main
from src.agents.clinical_agent import ClinicalAgent


class ClinicalAgentTests(unittest.TestCase):
    def test_run_uses_all_tools_and_requires_signoff(self):
        calls = []

        def generate_soap(transcript):
            calls.append(("soap", transcript))
            return {
                "subjective": "Fever and headache",
                "objective": "Vitals pending",
                "assessment": "Acute symptoms",
                "plan": "Clinician review",
            }

        def lookup_icd10(text):
            calls.append(("icd10", text))
            return [{"code": "R50.9", "description": "Fever, unspecified"}]

        async def readback(text, voice_gender):
            calls.append(("tts", text, voice_gender))
            return "https://audio.example/readback.wav"

        agent = ClinicalAgent(
            soap_generator=generate_soap,
            icd10_lookup=lookup_icd10,
            tts_readback=readback,
        )
        result = asyncio.run(agent.run("  Patient reports fever  ", voice_gender="female"))

        self.assertEqual([call[0] for call in calls], ["soap", "icd10", "tts"])
        self.assertIn("Fever and headache", calls[1][1])
        self.assertEqual(result["readback_audio_url"], "https://audio.example/readback.wav")
        self.assertTrue(result["requires_manual_review"])
        self.assertTrue(result["sign_off_required"])

    def test_local_icd10_lookup_returns_candidates(self):
        matches = ClinicalAgent._lookup_icd10("Patient reports fever and headache")

        self.assertEqual([match["code"] for match in matches], ["R50.9", "R51"])

    def test_run_can_skip_tts_readback(self):
        calls = []

        async def no_readback(*args):
            calls.append(args)

        agent = ClinicalAgent(
            soap_generator=lambda transcript: {"subjective": transcript},
            icd10_lookup=lambda text: [],
            tts_readback=no_readback,
        )
        result = asyncio.run(agent.run("Patient reports cough", include_readback=False))

        self.assertEqual(calls, [])
        self.assertNotIn("tts_readback", result["tools_used"])

    def test_run_rejects_empty_transcript(self):
        agent = ClinicalAgent()

        with self.assertRaises(ValueError):
            asyncio.run(agent.run("  "))

    def test_intron_soap_generation_posts_configured_request_and_parses_response(self):
        captured = {}
        provider_payload = {
            "subjective": "Patient reports fever.",
            "objective": "Not documented in transcript.",
            "assessment": "Fever reported; clinician assessment required.",
            "plan": "Review the source transcript.",
            "confidence_score": 0.5,
            "icd10_suggestions": [{"code": "R50.9", "description": "Fever, unspecified"}],
            "flagged_code_switches": [],
        }

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"data": provider_payload}

        class FakeAsyncClient:
            def __init__(self, *, timeout):
                captured["timeout"] = timeout

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

            async def post(self, url, *, headers=None, json=None):
                captured.update({"url": url, "headers": headers, "body": json})
                return FakeResponse()

        agent = ClinicalAgent(
            api_key="intron-test-key",
            endpoint="https://intron.example.test/clinical",
        )
        with patch("src.agents.clinical_agent.httpx.AsyncClient", FakeAsyncClient):
            soap = asyncio.run(agent._generate_soap("Patient reports fever."))

        self.assertEqual(soap["subjective"], provider_payload["subjective"])
        self.assertEqual(soap["icd10_codes"][0]["code"], "R50.9")
        self.assertEqual(captured["url"], "https://intron.example.test/clinical")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer intron-test-key")
        self.assertEqual(captured["timeout"], 5.0)
        self.assertEqual(captured["body"]["model"], "intron-sahara-v2.5")
        self.assertEqual(captured["body"]["language"], "auto")

    def test_intron_soap_generation_accepts_prefixed_bearer_key(self):
        captured = {}

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"data": {
                    "subjective": "Fever",
                    "objective": "Not documented",
                    "assessment": "Fever reported",
                    "plan": "Clinician review",
                }}

        class FakeAsyncClient:
            def __init__(self, *, timeout):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

            async def post(self, url, *, headers=None, json=None):
                captured["authorization"] = headers["Authorization"]
                return FakeResponse()

        agent = ClinicalAgent(api_key="Bearer intron-test-key", endpoint="https://intron.example.test/clinical")
        with patch("src.agents.clinical_agent.httpx.AsyncClient", FakeAsyncClient):
            asyncio.run(agent._generate_soap("fever"))

        self.assertEqual(captured["authorization"], "Bearer intron-test-key")

    def test_agent_route_scrubs_identifier_patterns_before_tool_call(self):
        async def exercise():
            soap = {
                "subjective": "Fever",
                "objective": "Not documented.",
                "assessment": "Fever reported.",
                "plan": "Clinician review.",
                "confidence_score": 0.5,
                "icd10_codes": [],
                "flagged_code_switches": [],
                "requires_manual_review": True,
                "requires_manual_entry": True,
            }
            with patch.object(main.clinical_agent, "run", new_callable=AsyncMock) as run_agent:
                run_agent.return_value = {
                    "soap": soap,
                    "generation_status": "generated",
                    "icd10_suggestions": [],
                    "readback_audio_url": None,
                    "tools_used": ["soap_generation", "icd10_lookup", "tts_readback"],
                }
                response = await main.process_clinical_agent(
                    main.ClinicalAgentRequest(
                        transcript="Patient test@example.org reports fever. Call 0912345678."
                    )
                )

            scrubbed = "Patient [REDACTED EMAIL] reports fever. Call [REDACTED PHONE]."
            run_agent.assert_awaited_once_with(
                scrubbed,
                voice_gender=main.INTRON_TTS_VOICE_GENDER,
                include_readback=True,
            )
            self.assertEqual(response["scrubbed_transcript"], scrubbed)
            self.assertEqual(response["redactions_count"], 2)
            self.assertTrue(response["sign_off_required"])

        asyncio.run(exercise())

    def test_agent_falls_back_with_local_candidates_without_intron_configuration(self):
        result = asyncio.run(
            ClinicalAgent(api_key="", endpoint="").run(
                "Patient reports fever and cough.", include_readback=False
            )
        )

        self.assertEqual(result["generation_status"], "fallback")
        self.assertEqual(result["provider"], "local_fallback")
        self.assertEqual([item["code"] for item in result["icd10_suggestions"]], ["R50.9", "R05"])
        self.assertTrue(result["soap"]["requires_manual_review"])
        self.assertTrue(result["sign_off_required"])

    def test_intron_timeout_and_http_errors_fall_back(self):
        class TimeoutClient:
            def __init__(self, *, timeout):
                self.timeout = timeout

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

            async def post(self, *args, **kwargs):
                import httpx

                raise httpx.TimeoutException("provider timed out")

        agent = ClinicalAgent(api_key="test-key", endpoint="https://intron.example.test/clinical")
        with patch("src.agents.clinical_agent.httpx.AsyncClient", TimeoutClient):
            result = asyncio.run(agent.run("Patient reports headache.", include_readback=False))

        self.assertEqual(result["generation_status"], "fallback")
        self.assertEqual(result["provider"], "local_fallback")
        self.assertIn("R51", [item["code"] for item in result["icd10_suggestions"]])
        self.assertEqual(result["provider_error"], "Intron clinical NLP request timed out.")

    def test_external_nlp_404_fallback_explains_endpoint_configuration(self):
        class NotFoundClient:
            def __init__(self, *, timeout):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

            async def post(self, url, **kwargs):
                import httpx

                request = httpx.Request("POST", url)
                response = httpx.Response(404, request=request)
                raise httpx.HTTPStatusError("Not found", request=request, response=response)

        agent = ClinicalAgent(api_key="test-key", endpoint="https://intron.example.test/soap")
        with patch("src.agents.clinical_agent.httpx.AsyncClient", NotFoundClient):
            result = asyncio.run(agent.run("Patient reports fever.", include_readback=False))

        self.assertEqual(result["generation_status"], "fallback")
        self.assertIn("HTTP 404", result["provider_error"])
        self.assertIn("endpoint path", result["provider_error"])


if __name__ == "__main__":
    unittest.main()