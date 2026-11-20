# Security

## Reporting a problem

Please report a vulnerability privately through GitHub: **Security tab, Report a vulnerability** on this repository. Do not open a public issue for it. I will reply as soon as I can and credit you if you want.

## What is and is not exposed

- **The website** (`index.html` and the `.js` files) is static. It runs entirely in the visitor's browser, stores nothing on a server, and has no accounts or secrets. Results you add from the runner stay in your browser.
- **The optional backend** (`backend/`) is self-hosted. The public read endpoints are open. Anything that changes data (importing a reference run, entering a price) needs `OMEN_ADMIN_KEY`; if it is not set, nobody can.
- **Running it in production:** set `OMEN_ENV=production`, `OMEN_CORS` (your site's origin), `OMEN_ADMIN_KEY` and `POSTGRES_PASSWORD` as secrets on the host, never in the repository. Put a reverse proxy with TLS, request-size and rate limits in front of it.

## Hardening already in the code

- Admin and API keys are compared in constant time; request bodies over 64 kB (including chunked ones) are rejected.
- Responses carry `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy` and `Cache-Control: no-store`.
- Production mode refuses to start with a wildcard CORS setting.
- Pasted runner results are checked against the catalog, escaped before display, and source links must be http(s).
- The container runs as a non-root user. Dependabot watches the Python, Docker and GitHub Actions dependencies.

## Known limits

- The admin key has no lockout after repeated wrong guesses; use a long random key and rate-limit at the proxy.
- The website has no Content-Security-Policy yet, because it uses inline scripts and also opens from `file://`.
