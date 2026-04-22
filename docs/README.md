# docs/ — static test client for the reise-API

`index.html` is a self-contained single-page chat UI that calls the Anvil
app's `/ask` endpoint and renders the answer + cited and related FHI
articles. Open it directly in a browser (no build step) — it stores the
API key, API URL, and optional region filter in localStorage.

Default API URL: `https://reiseraad.anvil.app/_/api/ask`.

This file is intentionally kept inside the anvil_repo so it ships alongside
the server code; host it from GitHub Pages (repo Settings → Pages →
`main` / `/docs`) for a shareable tester page.
