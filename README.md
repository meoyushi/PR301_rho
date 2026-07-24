# rho — Résumé Holistic Optimizer

Provenance-verified résumé parsing, ATS-calibrated scoring, and gated rewriting,
with a Next.js résumé studio on top. This runs the Gemini-backed backend and the
frontend together in Docker.

## Run with Docker (Gemini backend + frontend)

**1. Provide a Gemini API key.** Create a `.env` at the repo root:

```
GEMINI_API_KEY=your-key-here
```

The value may be a single key or a JSON array of keys (the client rotates across
them to spread free-tier quota). The key is read at container start and written
to a `.env` inside the backend container — it is never baked into the image.

**2. Bring the stack up:**

```bash
docker compose up --build
```

**3. Open the app:** http://localhost:3000  (backend API: http://localhost:8000,
docs at http://localhost:8000/docs).

The browser calls the backend directly at `http://localhost:8000`, so that URL is
baked into the frontend bundle at build time. To point the frontend at a
different backend address, set `NEXT_PUBLIC_API_URL` before building:

```bash
NEXT_PUBLIC_API_URL=http://192.168.1.50:8000 docker compose up --build
```

### Notes

- **First request is slow.** Sentence-transformer weights download on first use
  and are cached in the `rho-models` volume, so subsequent runs are fast.
- **Ports:** frontend `3000`, backend `8000`. Change the left-hand side of each
  `ports:` mapping in `docker-compose.yml` to remap on the host.
- **Backend only:**
  `docker build -f Dockerfile.backend -t rho-backend . && docker run --rm -p 8000:8000 -e GEMINI_API_KEY=... rho-backend`
- **Local (no Docker):** run the backend with
  `extraction_backend=gemini uvicorn rho.api.app:app --port 8000` and the
  frontend with `cd web && npm run dev`.

## Files

- `Dockerfile.backend` — FastAPI backend (Gemini extraction/rewrite + DOCX export).
- `web/Dockerfile` — Next.js frontend (standalone build).
- `docker-compose.yml` — wires both together.
- `docker/backend-entrypoint.sh` — materialises `.env` from `GEMINI_API_KEY`.
