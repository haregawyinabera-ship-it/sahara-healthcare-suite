import asyncio
import base64
import io
import json
import unittest
import wave
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import parse_qs, urlsplit

import config
from fastapi import HTTPException, UploadFile
from starlette.responses import PlainTextResponse
from starlette.requests import Request

import main


class SafetyTests(unittest.TestCase):
    def test_intron_sahara_key_is_supported_as_an_alias(self):
        with patch.dict("os.environ", {"INTRON_API_KEY": "", "INTRON_SAHARA_API_KEY": "sahara-key"}):
            self.assertEqual(config.get_intron_api_key(), "sahara-key")

    def test_intron_api_key_takes_precedence_over_alias(self):
        with patch.dict("os.environ", {"INTRON_API_KEY": "primary-key", "INTRON_SAHARA_API_KEY": "alias-key"}):
            self.assertEqual(config.get_intron_api_key(), "primary-key")

    def test_negated_maternal_symptom_does_not_escalate(self):
        result = main._classify_maternal_acuity("The patient reports no heavy bleeding.")
        self.assertEqual(result["level"], 5)
        self.assertFalse(result["is_emergency_trigger"])

    def test_positive_maternal_symptom_still_escalates(self):
        result = main._classify_maternal_acuity("The patient has heavy bleeding.")
        self.assertEqual(result["level"], 1)
        self.assertTrue(result["is_emergency_trigger"])

    def test_missing_provider_key_fails_closed(self):
        original_key = main.INTRON_API_KEY
        try:
            main.INTRON_API_KEY = ""
            request = Mock()
            upload = UploadFile(file=io.BytesIO(b"audio"), filename="test.wav")
            with self.assertRaises(HTTPException) as context:
                asyncio.run(main.transcribe_audio(request, upload))
            self.assertEqual(context.exception.status_code, 503)
        finally:
            main.INTRON_API_KEY = original_key

    def test_gemini_api_key_is_sent_in_header_not_url(self):
        captured = {}

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"candidates": [{"content": {"parts": [{"text": "transcript"}]}}]}

        class FakeAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

            async def post(self, url, *, headers=None, json=None):
                captured.update({"url": url, "headers": headers, "body": json})
                return FakeResponse()

        original_keys = (main.GEMINI_API_KEY, main.GEMINI_API_KEY_1, main.GEMINI_API_KEY_2)
        try:
            main.GEMINI_API_KEY = "test-gemini-key"
            main.GEMINI_API_KEY_1 = ""
            main.GEMINI_API_KEY_2 = ""
            with patch("main.httpx.AsyncClient", return_value=FakeAsyncClient()):
                transcript = asyncio.run(main._benchmark_gemini(b"audio", "audio/wav"))
        finally:
            main.GEMINI_API_KEY, main.GEMINI_API_KEY_1, main.GEMINI_API_KEY_2 = original_keys

        self.assertEqual(transcript, "transcript")
        self.assertNotIn("test-gemini-key", captured["url"])
        self.assertEqual(captured["headers"]["x-goog-api-key"], "test-gemini-key")

    def test_gemini_rotates_to_next_key_after_429(self):
        used_keys = []

        class FakeResponse:
            def __init__(self, status_code, transcript=""):
                self.status_code = status_code
                self.transcript = transcript

            def raise_for_status(self):
                if self.status_code >= 400:
                    request = main.httpx.Request("POST", "https://generativelanguage.googleapis.com")
                    response = main.httpx.Response(self.status_code, request=request)
                    raise main.httpx.HTTPStatusError("rate limited", request=request, response=response)

            def json(self):
                return {"candidates": [{"content": {"parts": [{"text": self.transcript}]}}]}

        class FakeAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

            async def post(self, url, *, headers=None, json=None):
                used_keys.append(headers["x-goog-api-key"])
                if len(used_keys) == 1:
                    return FakeResponse(429)
                return FakeResponse(200, "fallback transcript")

        original_keys = (main.GEMINI_API_KEY, main.GEMINI_API_KEY_1, main.GEMINI_API_KEY_2)
        try:
            main.GEMINI_API_KEY = "key-slot-one"
            main.GEMINI_API_KEY_1 = "key-slot-two"
            main.GEMINI_API_KEY_2 = ""
            with patch("main.httpx.AsyncClient", return_value=FakeAsyncClient()):
                transcript = asyncio.run(main._benchmark_gemini(b"audio", "audio/wav"))
        finally:
            main.GEMINI_API_KEY, main.GEMINI_API_KEY_1, main.GEMINI_API_KEY_2 = original_keys

        self.assertEqual(transcript, "fallback transcript")
        self.assertEqual(used_keys, ["key-slot-one", "key-slot-two"])

    def test_gemini_exhausted_key_pool_marks_429_non_retryable(self):
        class FakeResponse:
            status_code = 429

            def raise_for_status(self):
                request = main.httpx.Request("POST", "https://generativelanguage.googleapis.com")
                response = main.httpx.Response(429, request=request)
                raise main.httpx.HTTPStatusError("rate limited", request=request, response=response)

        class FakeAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return None

            async def post(self, url, *, headers=None, json=None):
                return FakeResponse()

        original_keys = (main.GEMINI_API_KEY, main.GEMINI_API_KEY_1, main.GEMINI_API_KEY_2)
        try:
            main.GEMINI_API_KEY = "key-slot-one"
            main.GEMINI_API_KEY_1 = "key-slot-two"
            main.GEMINI_API_KEY_2 = ""
            with patch("main.httpx.AsyncClient", return_value=FakeAsyncClient()):
                with self.assertRaises(main.GeminiKeysRateLimitedError) as context:
                    asyncio.run(main._benchmark_gemini(b"audio", "audio/wav"))
        finally:
            main.GEMINI_API_KEY, main.GEMINI_API_KEY_1, main.GEMINI_API_KEY_2 = original_keys

        self.assertEqual(context.exception.response.status_code, 429)
        self.assertTrue(context.exception.stop_retries)

    def test_upload_limit_is_enforced(self):
        upload = UploadFile(
            file=io.BytesIO(b"x" * (main.MAX_AUDIO_BYTES + 1)),
            filename="too-large.wav",
        )
        with self.assertRaises(HTTPException) as context:
            asyncio.run(main._read_limited_upload(upload))
        self.assertEqual(context.exception.status_code, 413)

    def test_wav_duration_limit_is_enforced(self):
        wav_data = io.BytesIO()
        with wave.open(wav_data, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(1)
            audio.writeframes(b"\0\0" * 121)

        with self.assertRaises(HTTPException) as context:
            main._validate_wav_duration(wav_data.getvalue())
        self.assertEqual(context.exception.status_code, 413)

    def test_stt_stream_url_uses_documented_endpoint_and_configuration(self):
        url = main._build_intron_stt_stream_url({"use_language_asr_input": "en"})
        parsed = urlsplit(url)
        self.assertEqual(parsed.scheme, "wss")
        self.assertEqual(parsed.netloc, "infer.voice.intron.io")
        self.assertEqual(parsed.path, "/stt/v1/stream")
        self.assertEqual(
            parse_qs(parsed.query),
            {
                "sample_rate": ["16000"],
                "bit_rate": ["16"],
                "num_channels": ["1"],
                "use_language_asr_input": ["en"],
            },
        )

    def test_stt_stream_rejects_audio_config_not_supported_by_browser(self):
        with self.assertRaises(ValueError):
            main._build_intron_stt_stream_url({"sample_rate": "48000"})

    def test_stt_websocket_logs_disallowed_origin(self):
        async def exercise():
            browser = Mock()
            browser.headers = {"origin": "https://unexpected.example"}
            browser.query_params = {}
            browser.close = AsyncMock()
            with (
                patch.object(main, "ALLOWED_ORIGINS", ["https://sahara-healthcare-suite.pages.dev"]),
                patch.object(main.logger, "warning") as warning,
            ):
                await main.websocket_stream(browser)
            browser.close.assert_awaited_once_with(code=1008)
            warning.assert_called_once_with("Rejecting STT WebSocket: origin is not allowed")

        asyncio.run(exercise())

    def test_stt_websocket_logs_missing_proxy_identity(self):
        async def exercise():
            browser = Mock()
            browser.headers = {"origin": "https://sahara-healthcare-suite.pages.dev"}
            browser.query_params = {}
            browser.close = AsyncMock()
            with (
                patch.object(main, "ALLOWED_ORIGINS", ["https://sahara-healthcare-suite.pages.dev"]),
                patch.object(main, "REQUIRE_PROXY_AUTH", True),
                patch.object(main.logger, "warning") as warning,
            ):
                await main.websocket_stream(browser)
            browser.close.assert_awaited_once_with(code=1008)
            warning.assert_called_once_with("Rejecting STT WebSocket: proxy identity is missing")

        asyncio.run(exercise())

    def test_stt_websocket_logs_missing_intron_key(self):
        async def exercise():
            browser = Mock()
            browser.headers = {"origin": "https://sahara-healthcare-suite.pages.dev"}
            browser.query_params = {}
            browser.close = AsyncMock()
            with (
                patch.object(main, "ALLOWED_ORIGINS", ["https://sahara-healthcare-suite.pages.dev"]),
                patch.object(main, "REQUIRE_PROXY_AUTH", False),
                patch.object(main, "INTRON_API_KEY", ""),
                patch.object(main.logger, "error") as error,
            ):
                await main.websocket_stream(browser)
            browser.close.assert_awaited_once_with(code=1011)
            error.assert_called_once_with("Rejecting STT WebSocket: INTRON_API_KEY is not configured")

        asyncio.run(exercise())

    def test_stt_audio_chunks_meet_provider_size_limits(self):
        small_audio = bytearray(b"\x01\x00" * 300)
        self.assertIsNone(main._take_stt_audio_chunk(small_audio))
        padded_chunk = main._take_stt_audio_chunk(small_audio, final=True)
        self.assertEqual(len(padded_chunk), main.STT_STREAM_MIN_CHUNK_BYTES)
        self.assertEqual(padded_chunk[:600], b"\x01\x00" * 300)
        self.assertFalse(small_audio)

        large_audio = bytearray(main.STT_STREAM_MAX_CHUNK_BYTES + 2)
        first_chunk = main._take_stt_audio_chunk(large_audio)
        final_chunk = main._take_stt_audio_chunk(large_audio, final=True)
        self.assertEqual(len(first_chunk), main.STT_STREAM_MAX_CHUNK_BYTES)
        self.assertEqual(len(final_chunk), main.STT_STREAM_MIN_CHUNK_BYTES)

    def test_stt_websocket_translates_audio_ack_and_commit_messages(self):
        class BrowserSocket:
            def __init__(self):
                self.headers = {"origin": "http://localhost:3000"}
                self.query_params = {"use_language_asr_input": "en"}
                self.incoming = asyncio.Queue()
                self.outgoing = []

            async def accept(self):
                pass

            async def receive(self):
                return await self.incoming.get()

            async def send_json(self, payload):
                self.outgoing.append(payload)

            async def close(self, code=1000, reason=None):
                pass

        class IntronSocket:
            def __init__(self):
                self.incoming = asyncio.Queue()
                self.sent = []

            async def __aenter__(self):
                await self.incoming.put(json.dumps({
                    "message_type": "SESSION_CREATED",
                    "session_id": "provider-session",
                }))
                return self

            async def __aexit__(self, exc_type, exc_value, traceback):
                return False

            async def send(self, raw_message):
                payload = json.loads(raw_message)
                self.sent.append(payload)
                if payload["message_type"] == "INPUT_AUDIO_CHUNK":
                    await self.incoming.put(json.dumps({
                        "message_type": "AUDIO_CHUNK_ACK",
                        "chunk_id": payload["ack_id"],
                    }))
                elif payload["message_type"] == "COMMIT":
                    await self.incoming.put(json.dumps({
                        "message_type": "COMMITTED_TRANSCRIPT",
                        "transcript_text": "test transcript",
                    }))

            def __aiter__(self):
                return self

            async def __anext__(self):
                return await self.incoming.get()

        async def exercise():
            browser = BrowserSocket()
            first_audio = b"\x01\x00" * 300
            second_audio = b"\x02\x00" * 212
            for event in (
                {"type": "websocket.receive", "text": json.dumps({"type": "audio_meta", "timestamp_ms": 1000})},
                {"type": "websocket.receive", "bytes": first_audio},
                {"type": "websocket.receive", "text": json.dumps({"type": "audio_meta", "timestamp_ms": 1100})},
                {"type": "websocket.receive", "bytes": second_audio},
                {"type": "websocket.receive", "text": json.dumps({"event": "stop"})},
            ):
                await browser.incoming.put(event)

            intron = IntronSocket()
            with (
                patch.object(main, "ALLOWED_ORIGINS", ["http://localhost:3000"]),
                patch.object(main, "REQUIRE_PROXY_AUTH", False),
                patch.object(main, "INTRON_API_KEY", "test-key"),
                patch.object(main.persistence, "start_session", new_callable=AsyncMock, return_value="local-session"),
                patch.object(main.persistence, "record_transcript", new_callable=AsyncMock) as record_transcript,
                  patch.object(main.persistence, "end_session", new_callable=AsyncMock) as end_session,
                patch.object(main.websockets, "connect", return_value=intron),
            ):
                await main.websocket_stream(browser)

            audio_messages = [message for message in intron.sent if message["message_type"] == "INPUT_AUDIO_CHUNK"]
            self.assertEqual(len(audio_messages), 1)
            self.assertEqual(audio_messages[0]["ack_id"], 1)
            self.assertEqual(
                base64.b64decode(audio_messages[0]["audio_base_64"]),
                first_audio + second_audio,
            )
            self.assertEqual(intron.sent[-1], {"message_type": "COMMIT"})
            self.assertIn({"ack_ts": 1100}, browser.outgoing)
            transcript_messages = [
                message for message in browser.outgoing
                if message.get("transcript") == "test transcript"
            ]
            self.assertEqual(len(transcript_messages), 1)
            self.assertEqual(transcript_messages[0]["session_id"], "local-session")
            self.assertEqual(transcript_messages[0]["audio_timestamp_ms"], 1100)
            self.assertIsInstance(transcript_messages[0]["server_time_ms"], float)
            record_transcript.assert_not_awaited()
            end_session.assert_awaited_once_with("local-session")

        asyncio.run(exercise())

    def test_tts_text_limit_matches_documented_boundary(self):
        request = main.IntronTTSRequest(text="a" * 4096)
        self.assertEqual(len(request.text), 4096)
        with self.assertRaises(ValueError):
            main.IntronTTSRequest(text="a" * 4097)

    def test_tts_routes_use_documented_provider_urls(self):
        async def exercise():
            payload = main.IntronTTSRequest(text="Clinical reminder")
            for route, endpoint in (
                (main.intron_tts_generate, main.INTRON_TTS_ENDPOINT),
                (main.intron_tts_enqueue, main.INTRON_TTS_ENQUEUE_ENDPOINT),
            ):
                with patch.object(main, "_post_intron_json", new_callable=AsyncMock) as post:
                    post.return_value = {"status": "ok"}
                    result = await route(payload)
                    self.assertEqual(result, {"status": "ok"})
                    post.assert_awaited_once_with(endpoint, payload.model_dump())

        asyncio.run(exercise())

    def test_voicebot_workflow_uses_documented_provider_url(self):
        async def exercise():
            payload = {"name": "Reminder", "workflow_type": "ROBOCALL", "message": "Hello"}
            with patch.object(main, "_post_intron_json", new_callable=AsyncMock) as post:
                post.return_value = {"workflow_id": "workflow-1"}
                result = await main.create_intron_voicebot_workflow(payload)
                self.assertEqual(result, {"workflow_id": "workflow-1"})
                post.assert_awaited_once_with(main.INTRON_VOICEBOT_WORKFLOWS_ENDPOINT, payload)

        asyncio.run(exercise())

    def test_followup_session_pruning_removes_expired_sessions(self):
        original_sessions = main._followup_sessions.copy()
        try:
            main._followup_sessions.clear()
            main._followup_sessions["expired"] = {"last_access": 0}
            main._prune_followup_sessions()
            self.assertNotIn("expired", main._followup_sessions)
        finally:
            main._followup_sessions.clear()
            main._followup_sessions.update(original_sessions)

    def test_proxy_identity_header_is_recognized(self):
        self.assertTrue(main._has_proxy_identity({"x-authenticated-user": "clinician@example.org"}))
        self.assertTrue(main._has_proxy_identity({"cf-access-authenticated-user-email": "clinician@example.org"}))
        self.assertFalse(main._has_proxy_identity({}))

    def test_proxy_auth_middleware_rejects_missing_identity(self):
        original_setting = main.REQUIRE_PROXY_AUTH
        try:
            main.REQUIRE_PROXY_AUTH = True
            middleware = main.ProxyIdentityMiddleware(main.app)
            request = Request({"type": "http", "method": "POST", "path": "/api/v1/post-care/analyze", "headers": []})
            response = asyncio.run(middleware.dispatch(request, lambda _: PlainTextResponse("ok")))
            self.assertEqual(response.status_code, 401)
        finally:
            main.REQUIRE_PROXY_AUTH = original_setting

    def test_fhir_export_contains_patient_and_encounter_references(self):
        payload = main.FHIRExportRequest(
            patient_id="patient-1",
            encounter_id="encounter-1",
            chief_complaint="Chest pain",
            diagnosis_code="R07.9",
            diagnosis_display="Chest pain, unspecified",
        )
        bundle = asyncio.run(main.export_fhir(payload))
        resources = [entry["resource"] for entry in bundle["entry"]]
        self.assertEqual(bundle["resourceType"], "Bundle")
        self.assertEqual(resources[0]["id"], "patient-1")
        self.assertEqual(resources[1]["subject"]["reference"], "Patient/patient-1")
        self.assertEqual(resources[-1]["code"]["coding"][0]["code"], "R07.9")

    def test_fhir_export_contains_signed_soap_composition(self):
        payload = main.FHIRExportRequest(
            patient_id="patient-1",
            encounter_id="encounter-1",
            chief_complaint="Chest pain",
            soap=main.SOAPDraftPayload(
                transcript="Chest pain for two days.",
                subjective="Chest pain for two days.",
                objective="BP 140/90.",
                assessment="R07.9 - Chest pain, unspecified.",
                plan="Clinician review and follow-up.",
            ),
        )
        bundle = asyncio.run(main.export_fhir(payload))
        compositions = [
            entry["resource"]
            for entry in bundle["entry"]
            if entry["resource"]["resourceType"] == "Composition"
        ]
        self.assertEqual(len(compositions), 1)
        self.assertEqual(compositions[0]["section"][0]["text"]["div"], "Chest pain for two days.")

    def test_live_benchmark_reports_unconfigured_providers(self):
        upload = UploadFile(file=io.BytesIO(b"audio"), filename="sample.wav")
        with patch.object(main, "INTRON_API_KEY", ""):
            result = asyncio.run(
                main.live_benchmark(
                    upload,
                    "Patient has fever and cough.",
                    "am-ET",
                    "intron",
                    "verified",
                )
            )
        self.assertEqual(result["benchmark_type"], "live_provider_comparison")
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["model"], "Intron Sahara v2.5")
        self.assertEqual(result["results"][0]["status"], "unavailable")

    def test_live_benchmark_allows_transcript_only_mode(self):
        upload = UploadFile(file=io.BytesIO(b"audio"), filename="sample.wav")
        with patch.object(main, "INTRON_API_KEY", ""):
            result = asyncio.run(main.live_benchmark(upload, "", "am-ET", "intron", "verified"))
        self.assertEqual(result["scoring_status"], "transcript_only")

    def test_ehr_commit_fails_closed_without_endpoint(self):
        payload = main.EHRCommitRequest(
            patient_id="patient-1",
            encounter_id="encounter-1",
            chief_complaint="Chest pain",
        )
        original_endpoint = main.EHR_FHIR_ENDPOINT
        try:
            main.EHR_FHIR_ENDPOINT = ""
            with self.assertRaises(HTTPException) as context:
                asyncio.run(main.commit_to_ehr(payload))
            self.assertEqual(context.exception.status_code, 503)
        finally:
            main.EHR_FHIR_ENDPOINT = original_endpoint


if __name__ == "__main__":
    unittest.main()
