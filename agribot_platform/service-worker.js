const APP_CACHE = "agribot-platform-v9";
const SHELL = ["/", "/index.html", "/styles.css", "/i18n.js", "/app.js", "/manifest.webmanifest"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(APP_CACHE).then((cache) => cache.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((key) => key !== APP_CACHE).map((key) => caches.delete(key)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.pathname.startsWith("/api/")) {
    event.respondWith(
      fetch(event.request)
        // API responses are deliberately never cached here. The dashboard's
        // explicit localStorage snapshot is marked cached in the UI, while a
        // service-worker cache hit must never masquerade as live field data.
        .catch(() => new Response(JSON.stringify({ error: "offline" }), {
          status: 504,
          headers: { "Content-Type": "application/json", "X-Agribot-Offline": "1" },
        }))
    );
    return;
  }
  // Network-first keeps a newly deployed dashboard from being pinned to an
  // old shell; the cache remains an explicit offline fallback only.
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        const copy = response.clone();
        caches.open(APP_CACHE).then((cache) => cache.put(event.request, copy));
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});
