# AfriHealth AI deployment and operations

## Deploy the FastAPI backend to Hugging Face Spaces

This repository includes a Docker Space configuration (`Dockerfile` and the
Docker SDK metadata at the top of `README.md`). Create a Docker Space in your
Hugging Face account, then push or sync this repository to that Space. The
container listens on port `7860`; the static browser frontend remains on
Cloudflare Pages.

Add runtime secrets/variables in the Space settings, not in the repository:

```text
INTRON_API_KEY=your-rotated-intron-key
ALLOWED_ORIGINS=https://sahara-healthcare-suite.pages.dev
DATABASE_URL=postgresql://...
```

The Space container filesystem is ephemeral. Use an external PostgreSQL
database for production persistence, or configure an HF Storage Bucket mounted
at `/data` and set `DATABASE_URL=sqlite+aiosqlite:////data/edge_sync.sqlite3`
for non-production testing. Never rely on the container's default SQLite file
for durable production records.

After the Space is running, copy its public URL (for example,
`https://<owner>-<space-name>.hf.space`) and set it as
`window.SAHARA_API_ORIGIN` in `index.html`, replacing the current Railway
`productionApiOrigin` value. Because Cloudflare Pages serves this as a static
site, commit and redeploy the updated HTML after changing the Space URL. The
browser needs this API origin for REST and WebSocket requests. Confirm the
Space's CORS `ALLOWED_ORIGINS` exactly matches the Cloudflare Pages origin.

Verify `/health`, `/openapi.json`, `/api/v1/health/intron`, and
`/api/v1/agent/process` on the Space URL. Also test `/ws/stream` through the
Space URL, since audio transcription depends on WebSockets.

Do not handle real patient data on a public Space without an approved access
control layer, durable encrypted storage, and an appropriate privacy/compliance
review. `REQUIRE_PROXY_AUTH=true` requires a trusted gateway to inject one of
the configured identity headers; setting it without that gateway will reject
API calls. Use synthetic data while validating the deployment.

This prototype has two deployment surfaces:

1. a static browser frontend (`index.html`, JavaScript, and assets); and
2. a server-side API bridge (`main.py`) that keeps the Intron credential private
   and proxies speech requests.

The frontend may be hosted on Cloudflare Pages. The FastAPI service must run on
an HTTPS-capable host that supports WebSockets. Do not deploy the API key in
the static frontend.

## Recommended architecture

```text
Browser
  |
  | HTTPS / WSS
  v
Cloudflare Pages       Separate FastAPI service
static frontend  --->  /health
                       /api/intron/stt/upload-sync
                       /api/intron/tts/generate
                       /api/intron/tts/enqueue
                       /api/intron/voicebot/workflows
                      /api/v1/fhir/export
                      /api/v1/ehr/commit
                       /ws/stream
                              |
                              v
                       Intron Voice API
```

The browser records audio locally, presents playback for review, and uploads
only after the user explicitly selects the upload action. The API service
receives the audio and uses `INTRON_API_KEY` server-side.

## Required environment variables

Configure these only on the API host:

```text
INTRON_API_KEY=...
DATABASE_URL=${{Postgres.DATABASE_URL}}
ALLOWED_ORIGINS=https://your-project.pages.dev
REQUIRE_PROXY_AUTH=true
INTRON_TTS_VOICE_LANGUAGE=am
INTRON_TTS_VOICE_ACCENT=amharic
INTRON_TTS_VOICE_GENDER=female
```

Use a comma-separated list for multiple exact origins. Do not use `*` for
production CORS when credentials or protected clinical workflows are involved.
Configure `DATABASE_URL` to point to an external PostgreSQL database on the
selected API host. The backend converts a `postgresql://` URL to the asyncpg
SQLAlchemy driver automatically. For Hugging Face Spaces, see the Docker Space
steps above. The Cloudflare Pages API origin must be changed from Railway to the
Space URL. On Railway, the start command is
`uvicorn main:app --host 0.0.0.0 --port $PORT`.
Do not commit `.env` files, API keys, patient recordings, full transcripts, or
provider response IDs.

`/api/v1/fhir/export` always builds a server-side FHIR bundle. `/api/v1/ehr/commit`
returns `503` until `EHR_FHIR_ENDPOINT` is configured, and then submits the
bundle with the optional `EHR_API_KEY`.

For production, put the FastAPI service behind an identity-aware gateway such
as Cloudflare Access. Configure the gateway to strip incoming
`X-Authenticated-User` and `Cf-Access-Authenticated-User-Email` headers, then
inject one only after successful clinician authentication. Set
`REQUIRE_PROXY_AUTH=true` and firewall the FastAPI origin so it is reachable
only through that gateway. Origin checks alone are not user authentication.

The gateway exposes `POST /api/intron/stt/upload-sync` using the documented
`audio_file_blob` multipart field; `POST /api/intron/tts/generate` and
`POST /api/intron/tts/enqueue` accept JSON text up to 4,096 characters;
`/ws/intron/tts/stream` proxies TTS chunks of 10 to 100 characters with the
60-second idle and 300-second session limits; and
`POST /api/intron/voicebot/workflows` creates a workflow. The supplied provider
documentation does not specify a TTS job-status route, so the gateway does not
invent one.

## Run the API

Install the Python dependencies and start the service on an externally
reachable interface:

```bash
python -m pip install -r requirements.txt
export INTRON_API_KEY="your-key"
export ALLOWED_ORIGINS="https://your-project.pages.dev"
export REQUIRE_PROXY_AUTH="true"
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Terminate TLS at the hosting platform or reverse proxy. The public API URL
must be HTTPS, and WebSocket clients must connect through `wss://`.

## Deploy the frontend to Cloudflare Pages

The repository is a static site. In Cloudflare Pages, connect the repository
and use:

- **Build command:** none
- **Build output directory:** `.`
- **Root directory:** repository root

If the deployment platform requires a command, use a no-op command such as
`echo "static site"` rather than running a development server. Do not upload
the private `clinical_validation/` directory or any local audio artifacts.

Set the API origin in the deployed page before the application script loads:

```html
<script>
  window.SAHARA_API_ORIGIN = "https://sahara-healthcare-suite-production-e636.up.railway.app";
</script>
```

For a production deployment, this value should be injected during the
deployment process rather than edited manually for each environment.

## Post-deployment checks

1. Open `https://your-project.pages.dev/` and confirm the page loads.
2. Request `https://api.example.org/health` and confirm the service is online.
3. Confirm the API reports the expected Intron connection state.
4. Test one consented, de-identified recording:
   local playback -> explicit upload -> transcript -> clinician review.
5. Confirm browser developer tools show no API key in source, storage, or
   request bodies.
6. Confirm REST requests use HTTPS and WebSocket requests use WSS.
7. Confirm a request from an unapproved origin is rejected by CORS.
8. Review hosting logs and retention settings before handling real patient
   information.
9. Confirm unauthenticated HTTP and WebSocket requests are rejected when
   `REQUIRE_PROXY_AUTH=true`.

## Operational boundaries

This prototype is a clinician-reviewed documentation and decision-support aid.
Generated transcripts, entities, triage labels, coding suggestions, SOAP notes,
and medication suggestions require qualified human review. Clinical validation
materials are private evaluation and governance artifacts, not a public
approval claim. Use only consented, de-identified data for demonstrations and
evaluation.
