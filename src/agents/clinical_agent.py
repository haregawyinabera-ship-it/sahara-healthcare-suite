"""Tool orchestration for clinician-reviewed clinical documentation drafts."""

import inspect
import logging
import os
from typing import Any, Callable

import httpx


ICD10_CATALOG = (
    (("chest pain", "chest tightness"), "R07.9", "Chest pain, unspecified"),
    (("shortness of breath", "dyspnea", "breathing difficulty"), "R06.0", "Dyspnea"),
    (("fever", "pyrexia", "high temperature"), "R50.9", "Fever, unspecified"),
    (("headache",), "R51", "Headache"),
    (("cough",), "R05", "Cough"),
    (("abdominal pain", "stomach pain"), "R10.9", "Abdominal pain, unspecified"),
    (("diarrhea", "dehydration"), "A09", "Infectious gastroenteritis / diarrhea"),
    (("wound", "pus", "discharge", "surgical site"), "T81.4", "Infection following a procedure"),
    (("rash", "allergy", "itching"), "L29.9", "Pruritus, unspecified"),
)


Tool = Callable[..., Any]
logger = logging.getLogger(__name__)


class ClinicalAgent:
    """Run local SOAP drafting, ICD-10 lookup, and optional Intron TTS readback.

    Intron provides STT/TTS here; clinical text structuring uses the local
    processor. An explicit endpoint can opt into a compatible external service.
    """

    def __init__(
        self,
        *,
        soap_generator: Tool | None = None,
        icd10_lookup: Tool | None = None,
        tts_readback: Tool | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.api_key = (api_key if api_key is not None else os.getenv("INTRON_API_KEY", "")).strip()
        self.endpoint = (endpoint or "").strip()
        self.timeout_seconds = timeout_seconds
        self._uses_injected_soap_generator = soap_generator is not None
        self._uses_default_soap_generator = soap_generator is None and endpoint is None
        self.soap_generator = soap_generator or (
            self._generate_local_soap if self._uses_default_soap_generator else self._generate_soap
        )
        self.icd10_lookup = icd10_lookup or self._lookup_icd10
        self.tts_readback = tts_readback or self._generate_readback

    async def run(
        self,
        transcript: str,
        *,
        voice_gender: str = "female",
        include_readback: bool = True,
    ) -> dict[str, Any]:
        """Execute the clinical tools and return a draft requiring clinician sign-off."""
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError("transcript must be a non-empty string")

        generation_status = (
            "local"
            if self._uses_default_soap_generator
            else "injected" if self._uses_injected_soap_generator else "active"
        )
        provider_error = None
        try:
            soap = await self._call(self.soap_generator, transcript.strip())
        except Exception as error:
            logger.exception("Clinical SOAP generation failed; using manual-review fallback")
            soap = self._fallback_soap(transcript.strip())
            generation_status = "fallback"
            provider_error = self._provider_error_message(error)
        soap_data = self._serialize_soap(soap)
        lookup_text = " ".join(
            [transcript, *(str(soap_data.get(section, "")) for section in (
                "subjective", "objective", "assessment", "plan"
            ))]
        )
        suggestions = await self._call(self.icd10_lookup, lookup_text)
        suggestions = list(suggestions or [])

        tools_used = ["soap_generation", "icd10_lookup"]
        audio_url = None
        if include_readback:
            readback_text = "\n".join(
                f"{section.title()}: {soap_data.get(section, '')}"
                for section in ("subjective", "objective", "assessment", "plan")
            )
            try:
                audio_url = await self._call(self.tts_readback, readback_text, voice_gender)
            except Exception:
                logger.exception("Clinical SOAP readback failed")
            tools_used.append("tts_readback")

        return {
            "success": True,
            "soap": soap_data,
            "generation_status": generation_status,
            "provider": (
                "Intron Sahara v2.5"
                if generation_status == "active"
                else "local_clinical_processor"
                if generation_status == "local"
                else "local_fallback"
                if generation_status == "fallback"
                else "injected_tool"
            ),
            "provider_status": generation_status,
            "provider_error": provider_error,
            "icd10_suggestions": suggestions,
            "readback_audio_url": audio_url,
            "tools_used": tools_used,
            "requires_manual_review": True,
            "sign_off_required": True,
        }

    @staticmethod
    def _provider_error_message(error: Exception) -> str:
        if isinstance(error, httpx.HTTPStatusError):
            status_code = error.response.status_code
            if status_code == 404:
                return (
                    "Configured clinical NLP endpoint returned HTTP 404. Verify "
                    "the endpoint path with that provider's documentation."
                )
            if status_code in (401, 403):
                return f"Intron rejected the clinical NLP credentials (HTTP {status_code})."
            if status_code == 429:
                return "Intron clinical NLP rate limit reached (HTTP 429)."
            return f"Intron clinical NLP endpoint returned HTTP {status_code}."
        if isinstance(error, (httpx.TimeoutException, TimeoutError)):
            return "Intron clinical NLP request timed out."
        if isinstance(error, httpx.RequestError):
            return "Unable to connect to the Intron clinical NLP endpoint."
        if str(error) in (
            "INTRON_API_KEY is not configured",
            "Clinical NLP endpoint is not configured",
        ):
            return str(error)
        return f"Clinical SOAP generation failed ({type(error).__name__})."

    @staticmethod
    async def _call(tool: Tool, *args: Any) -> Any:
        result = tool(*args)
        if inspect.isawaitable(result):
            return await result
        return result

    @staticmethod
    def _serialize_soap(soap: Any) -> dict[str, Any]:
        if hasattr(soap, "model_dump"):
            return soap.model_dump()
        if isinstance(soap, dict):
            return soap
        raise TypeError("SOAP generation tool must return a model or dictionary")

    @staticmethod
    def _lookup_icd10(text: str) -> list[dict[str, Any]]:
        normalized = text.casefold()
        matches = []
        for terms, code, description in ICD10_CATALOG:
            matched_terms = [term for term in terms if term in normalized]
            if matched_terms:
                matches.append({
                    "code": code,
                    "description": description,
                    "matched_terms": matched_terms,
                })
        return matches

    async def _generate_soap(self, transcript: str) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("INTRON_API_KEY is not configured")
        if not self.endpoint:
            raise RuntimeError("Clinical NLP endpoint is not configured")

        authorization = self.api_key
        if not authorization.lower().startswith("bearer "):
            authorization = f"Bearer {authorization}"
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                self.endpoint,
                headers={"Authorization": authorization},
                json={
                    "model": "intron-sahara-v2.5",
                    "transcript": transcript,
                    "target_format": "SOAP",
                    "language": "auto",
                },
            )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
            payload = payload["data"]
        if not isinstance(payload, dict):
            raise ValueError("Intron clinical NLP response must be a JSON object")

        result = {
            "subjective": payload.get("subjective", ""),
            "objective": payload.get("objective", ""),
            "assessment": payload.get("assessment", ""),
            "plan": payload.get("plan", ""),
            "confidence_score": payload.get("confidence_score", 0.0),
            "icd10_codes": payload.get("icd10_suggestions", payload.get("icd10_codes", [])),
            "flagged_code_switches": payload.get("flagged_code_switches", []),
            "requires_manual_review": True,
            "requires_manual_entry": True,
        }
        if not all(isinstance(result[key], str) and result[key].strip() for key in (
            "subjective", "objective", "assessment", "plan"
        )):
            raise ValueError("Intron clinical NLP response is missing SOAP sections")
        import main

        return main._coerce_soap_payload(result).model_dump()

    @staticmethod
    async def _generate_local_soap(transcript: str) -> dict[str, Any]:
        """Use the app's deterministic clinical processor after Intron STT."""
        import main

        return main._fallback_soap(transcript, reason="local clinical processor").model_dump()

    @staticmethod
    def _fallback_soap(transcript: str) -> Any:
        import main

        return main.get_fallback_soap(transcript)

    @staticmethod
    async def _generate_readback(text: str, voice_gender: str) -> str | None:
        import main

        return await main._intron_tts_generate(text, voice_gender=voice_gender)