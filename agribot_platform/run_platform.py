#!/usr/bin/env python3
"""
Local Agribot field dashboard.

Serves organized inference data from agribot_inference_data over HTTP with no
external dependencies. The static UI caches the latest run in the phone browser,
which keeps the most recent data available after closing/reopening the page.
"""

from __future__ import annotations

import argparse
import csv
import hmac
import io
import ipaddress
import json
import math
import mimetypes
import os
import re
import threading
import socket
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from datetime import datetime


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "agribot_inference_data"
FARMER_CONFIG = PROJECT_ROOT / "farmer_config.json"
FIELD_LAYOUT_FILE = "field_layout.json"
APK_FILE = PROJECT_ROOT / "dist" / "agribot-field-app-debug.apk"
CONFIDENCE_THRESHOLD = 0.765
MAX_BODY_BYTES = 64 * 1024
MAX_JSON_LINE_BYTES = 1024 * 1024
MAX_CSV_BYTES = 16 * 1024 * 1024
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_FIELDS = 100
MAX_ROWS_PER_FIELD = 100
MAX_RUN_ID_LENGTH = 64
MAX_DECISION_LIMIT = 1000
MAX_QUERY_FIELDS = 32
MAX_REQUEST_SECONDS = 15
FORMULA_PREFIX_RE = re.compile(r"^[\t ]*[=+\-@]")
STATIC_FILES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/i18n.js": "i18n.js",
    "/app.js": "app.js",
    "/styles.css": "styles.css",
    "/manifest.webmanifest": "manifest.webmanifest",
    "/service-worker.js": "service-worker.js",
}


class RequestTooLarge(ValueError):
    """Raised when a request or response exceeds a hard safety bound."""


def reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def hmac_compare(left: str, right: str) -> bool:
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def json_loads_strict(text: str):
    return json.loads(text, parse_constant=reject_json_constant)


def finite_json(value):
    """Return a JSON-safe copy for legacy/inference data read from disk."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, tuple):
        return [finite_json(item) for item in value]
    return value


def validate_json_tree(value, *, depth: int = 0, max_depth: int = 12):
    """Reject unsafe JSON shapes before they reach layout normalization."""
    if depth > max_depth:
        raise ValueError("request JSON is too deeply nested")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite numeric value is not allowed")
    if isinstance(value, dict):
        if len(value) > 128:
            raise ValueError("request object has too many fields")
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 128:
                raise ValueError("request contains an unsafe field name")
            validate_json_tree(item, depth=depth + 1, max_depth=max_depth)
    elif isinstance(value, list):
        if len(value) > 1000:
            raise ValueError("request array is too large")
        for item in value:
            validate_json_tree(item, depth=depth + 1, max_depth=max_depth)
    elif isinstance(value, str) and len(value) > 100_000:
        raise ValueError("request string is too large")
    elif value is not None and not isinstance(value, (str, int, bool)):
        raise ValueError("request contains an unsupported JSON value")


def validate_layout_payload(payload: dict):
    """Validate the public field-layout schema before normalization."""
    validate_json_tree(payload, max_depth=8)
    if not isinstance(payload, dict):
        raise ValueError("layout must be an object")
    fields = payload.get("fields")
    field_items = fields if fields is not None else [payload]
    if not isinstance(field_items, list) or not field_items:
        raise ValueError("fields must be a non-empty array")
    if len(field_items) > MAX_FIELDS:
        raise ValueError(f"at most {MAX_FIELDS} fields are allowed")
    numeric_keys = {
        "row_count",
        "plants_per_row",
        "default_plants_per_row",
        "row_spacing_m",
        "plant_spacing_m",
        "start_plant",
        "plant_step",
        "plant_cooldown_sec",
    }
    text_keys = {"name", "field_name", "field_id", "id", "active_field_id", "active_row_id", "row_id"}
    for field in field_items:
        if not isinstance(field, dict):
            raise ValueError("each field must be an object")
        rows = field.get("rows")
        if rows is not None:
            if not isinstance(rows, list):
                raise ValueError("field rows must be an array")
            if len(rows) > MAX_ROWS_PER_FIELD:
                raise ValueError(f"at most {MAX_ROWS_PER_FIELD} rows are allowed per field")
        for key in numeric_keys:
            if key not in field or field[key] is None:
                continue
            value = field[key]
            if isinstance(value, bool):
                raise ValueError(f"{key} must be numeric")
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{key} must be numeric") from None
            if not math.isfinite(number):
                raise ValueError(f"{key} must be finite")
        for key in text_keys:
            if key in field and field[key] is not None and not isinstance(field[key], str):
                raise ValueError(f"{key} must be text")
        for row in rows or []:
            if not isinstance(row, dict):
                raise ValueError("each row must be an object")
            for key in numeric_keys:
                if key not in row or row[key] is None:
                    continue
                value = row[key]
                if isinstance(value, bool):
                    raise ValueError(f"{key} must be numeric")
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    raise ValueError(f"{key} must be numeric") from None
                if not math.isfinite(number):
                    raise ValueError(f"{key} must be finite")
    return payload


def safe_relative_path(root: Path, candidate: Path) -> Path:
    """Resolve a path and require it to remain below root."""
    root_resolved = root.expanduser().resolve()
    candidate_resolved = candidate.expanduser().resolve()
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("path escapes the configured data root") from exc
    return candidate_resolved


def csv_safe_cell(value) -> str:
    text = "" if value is None else str(value)
    if FORMULA_PREFIX_RE.match(text):
        return "'" + text
    return text


def finite_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def atomic_write_text(path: Path, text: str):
    """Write a small state file without exposing a partially-written JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a finite worker budget and daemon workers."""

    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True
    request_queue_size = 32

    def __init__(self, server_address, handler_class, *, max_workers: int = 8):
        super().__init__(server_address, handler_class)
        self._worker_slots = threading.BoundedSemaphore(max(1, min(int(max_workers), 32)))

    def process_request(self, request, client_address):
        if not self._worker_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._worker_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._worker_slots.release()

DEFAULT_FIELD_LAYOUT = {
    "active_field_id": "field_1",
    "fields": [],
    "field_id": "Field 1",
    "field_name": "Field 1",
    "active_row_id": "A",
    "row_count": 1,
    "plants_per_row": 100,
    "start_plant": 1,
    "plant_step": 1,
    "plant_cooldown_sec": 2.0,
    "row_spacing_m": 1.0,
    "plant_spacing_m": 0.5,
}

DEFAULT_FIELD = {
    "id": "field_1",
    "name": "Field 1",
    "active_row_id": "A",
    "row_count": 1,
    "plants_per_row": 100,
    "row_spacing_m": 1.0,
    "plant_spacing_m": 0.5,
    "start_plant": 1,
    "plant_step": 1,
    "plant_cooldown_sec": 2.0,
    "rows": [],
}


def now_ip() -> str:
    try:
        output = subprocess.check_output(["hostname", "-I"], text=True, timeout=0.5)
        for item in output.split():
            if item and not item.startswith("127."):
                return item
    except Exception:
        pass
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.2)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


def safe_run_id(run_id: str) -> str:
    value = unquote(run_id).strip()
    if (
        not value
        or len(value) > MAX_RUN_ID_LENGTH
        or value in {".", ".."}
        or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in value)
    ):
        raise ValueError("invalid run id")
    return value


def read_json(path: Path, default):
    try:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if len(text.encode("utf-8")) > MAX_JSON_LINE_BYTES:
                return default
            value = json_loads_strict(text)
            return finite_json(value)
    except Exception:
        return default
    return default


def read_jsonl(path: Path, *, limit: int = 200):
    if not path.exists():
        return []
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if len(line.encode("utf-8")) > MAX_JSON_LINE_BYTES:
                    continue
                try:
                    value = json_loads_strict(line)
                    if isinstance(value, dict):
                        rows.append(finite_json(value))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
    except Exception:
        return []
    if limit > 0:
        return rows[-limit:]
    return rows


def clamp_int(value, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return min(max(number, low), high)


def clamp_float(value, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return min(max(number, low), high)


def clean_text(value, default: str, *, limit: int = 80) -> str:
    text = "".join(ch for ch in str(value or "").strip() if ch >= " " and ch not in "\x7f\ufffe\uffff")
    if not text:
        return default
    return text[:limit]


def clean_id(value, default: str, *, limit: int = 48) -> str:
    text = str(value or "").strip().lower()
    cleaned = []
    last_sep = False
    for ch in text:
        if ch.isascii() and ch.isalnum():
            cleaned.append(ch)
            last_sep = False
        elif not last_sep:
            cleaned.append("_")
            last_sep = True
    result = "".join(cleaned).strip("_")[:limit]
    return result or default


def row_id_for_index(index: int) -> str:
    value = ""
    number = index
    while True:
        number, rem = divmod(number, 26)
        value = chr(65 + rem) + value
        if number == 0:
            return value
        number -= 1


def row_ids(count: int) -> list[str]:
    return [row_id_for_index(i) for i in range(max(1, count))]


def disease_location_summary(decisions: list[dict]) -> dict:
    summary = {}
    for row in decisions:
        label = str(row.get("label") or "Unknown")
        normalized = label.strip().lower().replace("_", " ")
        confidence = finite_float(row.get("confidence"), 0.0)
        if row.get("status") != "ok" or confidence < CONFIDENCE_THRESHOLD or normalized in {"healthy", "uncertain"}:
            continue
        entry = summary.setdefault(label, {"count": 0, "locations": []})
        entry["count"] += 1
        entry["locations"].append(
            {
                "field_id": row.get("field_id"),
                "row_id": row.get("row_id"),
                "plant_number": row.get("plant_number") or row.get("plant_id"),
                "plant_column": row.get("plant_column"),
                "confidence": finite_float(row.get("confidence"), 0.0),
                "x_m": finite_float(row.get("x_m"), 0.0),
                "y_m": finite_float(row.get("y_m"), 0.0),
            }
        )
    return summary


def max_plants_seen(decisions: list[dict]) -> int:
    max_plant = 0
    for row in decisions:
        try:
            max_plant = max(max_plant, int(row.get("plant_number") or row.get("plant_id") or row.get("sequence") or 0))
        except (TypeError, ValueError):
            continue
    return max_plant


class AgribotHandler(BaseHTTPRequestHandler):
    server_version = "AgribotPlatform/1.0"

    def setup(self):
        super().setup()
        self.request.settimeout(MAX_REQUEST_SECONDS)

    def origin_allowed(self, origin: str | None = None) -> bool:
        origin = origin if origin is not None else self.headers.get("Origin")
        if not origin:
            return True
        origin = origin.strip()
        configured = getattr(self.server, "allowed_origins", set())
        if origin in configured:
            return True
        parsed = urlparse(origin)
        host = (self.headers.get("Host") or "").strip().lower()
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or not host:
            return False
        return parsed.netloc.lower() == host

    def client_is_loopback(self) -> bool:
        try:
            return ipaddress.ip_address(self.client_address[0].split("%", 1)[0]).is_loopback
        except (ValueError, IndexError):
            return False

    def require_write_authorization(self) -> bool:
        """Require configured bearer auth and reject cross-origin writes."""
        origin = self.headers.get("Origin")
        if origin and not self.origin_allowed(origin):
            self.discard_request_body()
            self.send_json({"error": "origin is not allowed"}, status=403)
            return False
        configured_token = getattr(self.server, "auth_token", "")
        if configured_token:
            authorization = self.headers.get("Authorization") or ""
            scheme, _, provided = authorization.partition(" ")
            if scheme.lower() != "bearer" or not provided or not hmac_compare(provided.strip(), configured_token):
                self.discard_request_body()
                self.send_json({"error": "authorization required"}, status=401)
                return False
            return True
        # Without a configured token, preserve local/offline browser operation,
        # but do not accept unauthenticated cross-network writes.
        if not origin and not self.client_is_loopback():
            self.discard_request_body()
            self.send_json({"error": "same-origin request or configured token required"}, status=403)
            return False
        return True

    def discard_request_body(self):
        """Consume small rejected bodies so clients receive a clean HTTP error."""
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except (TypeError, ValueError):
            self.close_connection = True
            return
        if length <= 0:
            return
        to_read = min(length, MAX_BODY_BYTES * 2)
        try:
            self.rfile.read(to_read)
        except (OSError, TimeoutError):
            self.close_connection = True
        if length > to_read:
            self.close_connection = True

    @property
    def data_root(self) -> Path:
        return self.server.data_root

    def log_message(self, fmt, *args):
        if self.server.quiet:
            return
        super().log_message(fmt, *args)

    def end_headers(self):
        origin = self.headers.get("Origin")
        if origin and self.origin_allowed(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-CSRF-Token")
            self.send_header("Vary", "Origin")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; "
            "script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; worker-src 'self'",
        )
        super().end_headers()

    def do_OPTIONS(self):
        if self.headers.get("Origin") and not self.origin_allowed():
            self.send_json({"error": "origin is not allowed"}, status=403)
            return
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path.startswith("/api/"):
            try:
                query = parse_qs(parsed.query, max_num_fields=MAX_QUERY_FIELDS)
            except ValueError:
                self.send_json({"error": "query is too large"}, status=400)
                return
            self.handle_api(path, query)
            return
        self.handle_static(path)

    def do_POST(self):
        if not self.require_write_authorization():
            return
        parsed = urlparse(self.path)
        if parsed.path == "/api/farm-layout":
            self.handle_farm_layout_post()
            return
        if parsed.path == "/api/field-layout":
            self.handle_field_layout_post()
            return
        self.send_json({"error": "unknown endpoint"}, status=404)

    def do_HEAD(self):
        parsed = urlparse(self.path)
        self.handle_static(parsed.path, send_body=False)

    def handle_static(self, path: str, *, send_body: bool = True):
        if path == "/agribot-field-app-debug.apk":
            self.send_file(APK_FILE, "application/vnd.android.package-archive", send_body=send_body)
            return
        file_name = STATIC_FILES.get(path)
        if file_name is None:
            self.send_error(404, "Not found")
            return
        file_path = ROOT / file_name
        self.send_file(file_path, mimetypes.guess_type(str(file_path))[0] or "application/octet-stream", send_body=send_body)

    def send_file(self, file_path: Path, content_type: str, *, send_body: bool = True):
        try:
            resolved = file_path.resolve()
        except OSError:
            resolved = file_path
        if not resolved.is_file():
            self.send_error(404, "Missing static file")
            return
        if resolved.stat().st_size > MAX_RESPONSE_BYTES:
            self.send_error(413, "Static file too large")
            return
        data = resolved.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache" if resolved.suffix in {".html", ".js", ".apk", ".webmanifest"} else "public, max-age=3600")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if send_body:
            self.wfile.write(data)

    def send_json(self, payload, status: int = 200):
        data = json.dumps(finite_json(payload), indent=2, allow_nan=False).encode("utf-8")
        if len(data) > MAX_RESPONSE_BYTES:
            status = 500
            data = b'{"error":"response too large"}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_csv(self, path: Path, name: str):
        try:
            path = safe_relative_path(self.data_root / "runs", path)
        except ValueError:
            self.send_json({"error": "invalid csv path"}, status=400)
            return
        if not path.is_file():
            self.send_json({"error": "csv not found"}, status=404)
            return
        if path.stat().st_size > MAX_CSV_BYTES:
            self.send_json({"error": "csv too large"}, status=413)
            return
        try:
            raw = path.read_text(encoding="utf-8")
            csv.field_size_limit(1_000_000)
            output = io.StringIO(newline="")
            writer = csv.writer(output, lineterminator="\n")
            for row in csv.reader(io.StringIO(raw)):
                writer.writerow([csv_safe_cell(cell) for cell in row])
            data = output.getvalue().encode("utf-8")
        except (OSError, UnicodeError, csv.Error) as exc:
            self.log_message("CSV export failed: %s", exc)
            self.send_json({"error": "csv could not be read"}, status=422)
            return
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", name)[:120] or "agribot_export.csv"
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="{safe_name}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def handle_api(self, path: str, query: dict[str, list[str]]):
        try:
            if path == "/api/status":
                self.send_json(self.status_payload())
            elif path == "/api/farm-layout":
                self.send_json({"farm_layout": self.farm_layout_payload()})
            elif path == "/api/field-layout":
                self.send_json({"field_layout": self.field_layout_payload()})
            elif path == "/api/network-state":
                self.send_json({"network": self.network_state_payload()})
            elif path == "/api/runs":
                self.send_json({"runs": self.list_runs()})
            elif path == "/api/runs/latest":
                run_id = self.latest_run_id()
                self.send_json(self.run_payload(run_id))
            elif path.startswith("/api/runs/"):
                suffix = path.removeprefix("/api/runs/")
                parts = suffix.split("/")
                run_id = safe_run_id(parts[0])
                if len(parts) == 1:
                    self.send_json(self.run_payload(run_id))
                elif len(parts) == 2 and parts[1] == "decisions":
                    limit = clamp_int((query.get("limit") or ["200"])[0], 200, 1, MAX_DECISION_LIMIT)
                    self.send_json({"run_id": run_id, "decisions": self.run_decisions(run_id, limit=limit)})
                else:
                    self.send_json({"error": "unknown run endpoint"}, status=404)
            elif path.startswith("/api/download/") and path.endswith(".csv"):
                run_id = safe_run_id(path.removeprefix("/api/download/").removesuffix(".csv"))
                self.send_csv(self.run_dir(run_id) / "events.csv", f"agribot_{run_id}.csv")
            elif path == "/api/remote-monitor":
                self.send_json(self.remote_monitor_payload())
            else:
                self.send_json({"error": "unknown endpoint"}, status=404)
        except RequestTooLarge as exc:
            self.send_json({"error": str(exc)}, status=413)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self.log_message("API request failed: %s", exc)
            self.send_json({"error": "internal server error"}, status=500)

    def read_request_json(self, *, max_bytes: int = 8192):
        content_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise ValueError("Content-Type must be application/json")
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("chunked request bodies are not supported")
        try:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise ValueError("Content-Length is required")
            length = int(raw_length)
        except (TypeError, ValueError):
            raise ValueError("invalid Content-Length") from None
        if length <= 0:
            raise ValueError("request body is required")
        effective_max = min(max_bytes, MAX_BODY_BYTES)
        if length > effective_max:
            self.discard_request_body()
            raise RequestTooLarge("request body too large")
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("request body was truncated")
            payload = json_loads_strict(body.decode("utf-8"))
        except UnicodeDecodeError:
            raise ValueError("request body must be UTF-8 JSON") from None
        validate_json_tree(payload, max_depth=8)
        return payload

    def handle_field_layout_post(self):
        try:
            payload = validate_layout_payload(self.read_request_json(max_bytes=MAX_BODY_BYTES))
            existing = self.farm_layout_payload()
            if isinstance(payload, dict) and "fields" in payload:
                layout = self.normalize_farm_layout(payload, existing)
            else:
                active = self.normalize_field(payload, self.field_layout_payload())
                layout = self.farm_layout_payload()
                fields = []
                replaced = False
                for field in layout["fields"]:
                    if field["id"] == active["id"]:
                        fields.append(active)
                        replaced = True
                    else:
                        fields.append(field)
                if not replaced:
                    fields.append(active)
                layout["fields"] = fields
                layout["active_field_id"] = active["id"]
                layout["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            path = safe_relative_path(self.data_root, self.data_root / FIELD_LAYOUT_FILE)
            with self.server.write_lock:
                atomic_write_text(path, json.dumps(layout, indent=2, allow_nan=False) + "\n")
                self.update_farmer_config(layout)
            self.send_json({"ok": True, "farm_layout": layout, "field_layout": self.active_field_layout(layout)})
        except RequestTooLarge as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=413)
        except ValueError as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            self.log_message("field layout update failed: %s", exc)
            self.send_json({"ok": False, "error": "internal server error"}, status=500)

    def handle_farm_layout_post(self):
        try:
            payload = validate_layout_payload(self.read_request_json(max_bytes=MAX_BODY_BYTES))
            layout = self.normalize_farm_layout(payload, self.farm_layout_payload())
            layout["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            path = safe_relative_path(self.data_root, self.data_root / FIELD_LAYOUT_FILE)
            with self.server.write_lock:
                atomic_write_text(path, json.dumps(layout, indent=2, allow_nan=False) + "\n")
                self.update_farmer_config(layout)
            self.send_json({"ok": True, "farm_layout": layout, "field_layout": self.active_field_layout(layout)})
        except RequestTooLarge as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=413)
        except ValueError as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            self.log_message("farm layout update failed: %s", exc)
            self.send_json({"ok": False, "error": "internal server error"}, status=500)

    def remote_monitor_payload(self):
        """Simplified status for remote monitoring (e.g., SMS alerts, external dashboards)."""
        runs = self.list_runs()
        latest = self.latest_run_id()
        latest_summary = self.run_payload(latest)["summary"] if latest else {}
        field_summaries = self.field_summaries(
            self.farm_layout_payload(
                metadata=self.run_payload(latest).get("metadata", {}) if latest else {},
                decisions=self.run_decisions(latest, limit=1000) if latest else [],
            ),
            self.run_decisions(latest, limit=1000) if latest else [],
        ) if latest else []
        total_sick = sum(fs.get("sick", 0) for fs in field_summaries)
        total_uncertain = sum(fs.get("uncertain", 0) for fs in field_summaries)
        return {
            "ok": True,
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "latest_run_id": latest,
            "total_runs": len(runs),
            "latest_summary": latest_summary,
            "fields": [
                {
                    "field_id": fs.get("field_id"),
                    "field_name": fs.get("field_name"),
                    "total_detected": fs.get("total_detected", 0),
                    "sick": fs.get("sick", 0),
                    "uncertain": fs.get("uncertain", 0),
                    "disease_count": len(fs.get("diseases", {})),
                }
                for fs in field_summaries
            ],
            "total_sick": total_sick,
            "total_uncertain": total_uncertain,
            "alert": total_sick > 0 or total_uncertain > 0,
        }

    def status_payload(self):
        runs = self.list_runs()
        latest = self.latest_run_id()
        return {
            "ok": True,
            "data_root": str(self.data_root),
            "run_count": len(runs),
            "latest_run_id": latest,
            "latest": self.run_payload(latest)["summary"] if latest else None,
            "farm_layout": self.farm_layout_payload(),
            "field_layout": self.field_layout_payload(),
            "network": self.network_state_payload(),
        }

    def network_state_payload(self):
        state = read_json(self.data_root / "network_state.json", {})
        if not isinstance(state, dict):
            state = {}
        return {
            "mode": state.get("mode") or "unknown",
            "dashboard_url": state.get("dashboard_url") or f"http://{now_ip()}:{self.server.server_port}",
            "apk_url": state.get("apk_url"),
            "wifi_ssid": state.get("wifi_ssid"),
            "hotspot_ssid": state.get("hotspot_ssid"),
            "updated_at": state.get("updated_at"),
        }

    def run_dir(self, run_id: str) -> Path:
        safe_id = safe_run_id(run_id)
        return safe_relative_path(self.data_root / "runs", self.data_root / "runs" / safe_id)

    def latest_run_id(self):
        latest = read_json(self.data_root / "latest_run.json", {})
        run_id = latest.get("run_id")
        if run_id:
            try:
                return safe_run_id(str(run_id))
            except ValueError:
                pass
        runs = self.list_runs()
        return runs[0]["run_id"] if runs else None

    def list_runs(self):
        runs_root = self.data_root / "runs"
        if not runs_root.exists():
            return []
        rows = []
        for run_dir in runs_root.iterdir():
            if not run_dir.is_dir():
                continue
            try:
                run_id = safe_run_id(run_dir.name)
            except ValueError:
                continue
            summary = read_json(run_dir / "summary.json", {})
            metadata = read_json(run_dir / "metadata.json", {})
            rows.append(
                {
                    "run_id": run_id,
                    "started_at": summary.get("started_at") or metadata.get("started_at"),
                    "last_updated_at": summary.get("last_updated_at") or metadata.get("completed_at"),
                    "completed": bool(summary.get("completed")),
                    "decisions": clamp_int(summary.get("decisions"), 0, 0, 10_000_000),
                    "ok": clamp_int(summary.get("ok"), 0, 0, 10_000_000),
                    "uncertain": clamp_int(summary.get("uncertain"), 0, 0, 10_000_000),
                }
            )
        rows.sort(key=lambda item: item.get("last_updated_at") or item.get("started_at") or "", reverse=True)
        return rows

    def run_payload(self, run_id: str | None):
        if not run_id:
            farm_layout = self.farm_layout_payload()
            return {
                "run_id": None,
                "metadata": {},
                "summary": {},
                "decisions": [],
                "farm_layout": farm_layout,
                "field_layout": self.active_field_layout(farm_layout),
                "field_summaries": [],
            }
        run_dir = self.run_dir(run_id)
        metadata = read_json(run_dir / "metadata.json", {})
        summary = read_json(run_dir / "summary.json", {})
        decisions = self.run_decisions(run_id, limit=1000)
        farm_layout = self.farm_layout_payload(metadata=metadata, decisions=decisions)
        return {
            "run_id": run_id,
            "metadata": metadata,
            "summary": summary,
            "decisions": decisions,
            "farm_layout": farm_layout,
            "field_layout": self.active_field_layout(farm_layout),
            "field_summaries": self.field_summaries(farm_layout, decisions),
            "csv_url": f"/api/download/{run_id}.csv",
        }

    def run_decisions(self, run_id: str, *, limit: int = 200):
        return read_jsonl(self.run_dir(run_id) / "decisions.jsonl", limit=limit)

    def farm_layout_payload(self, *, metadata: dict | None = None, decisions: list[dict] | None = None):
        metadata = metadata or {}
        decisions = decisions or []
        saved = read_json(self.data_root / FIELD_LAYOUT_FILE, {})
        field_map = metadata.get("field_map") or {}
        base = dict(saved) if isinstance(saved, dict) else {}
        if "fields" not in base:
            base = {
                **base,
                "active_field_id": clean_id(base.get("active_field_id") or base.get("field_id") or field_map.get("field_id"), "field_1"),
                "fields": [
                    {
                        **DEFAULT_FIELD,
                        "id": clean_id(base.get("active_field_id") or base.get("field_id") or field_map.get("field_id"), "field_1"),
                        "name": base.get("field_id") or field_map.get("field_id") or DEFAULT_FIELD["name"],
                        "active_row_id": base.get("active_row_id") or field_map.get("active_row_id") or field_map.get("row_id") or DEFAULT_FIELD["active_row_id"],
                        "row_count": base.get("row_count") or field_map.get("row_count") or DEFAULT_FIELD["row_count"],
                        "plants_per_row": base.get("plants_per_row") or field_map.get("plants_per_row") or DEFAULT_FIELD["plants_per_row"],
                        "row_spacing_m": base.get("row_spacing_m") or field_map.get("row_spacing_m") or DEFAULT_FIELD["row_spacing_m"],
                        "plant_spacing_m": base.get("plant_spacing_m") or field_map.get("plant_spacing_m") or DEFAULT_FIELD["plant_spacing_m"],
                        "start_plant": base.get("start_plant") or field_map.get("start_plant") or DEFAULT_FIELD["start_plant"],
                        "plant_step": base.get("plant_step") or field_map.get("plant_step") or DEFAULT_FIELD["plant_step"],
                        "plant_cooldown_sec": base.get("plant_cooldown_sec") or field_map.get("plant_cooldown_sec") or DEFAULT_FIELD["plant_cooldown_sec"],
                        "rows": base.get("rows") or field_map.get("rows") or [],
                    }
                ],
            }
        elif field_map and not saved:
            base.setdefault("active_field_id", clean_id(field_map.get("field_id"), "field_1"))

        layout = self.normalize_farm_layout(base, DEFAULT_FIELD_LAYOUT)

        # If live metadata refers to a field not yet saved by the UI, add it so
        # old runs and fresh camera runs still appear in the farmer platform.
        live_name = field_map.get("field_id")
        if live_name:
            live_id = clean_id(field_map.get("active_field_id") or live_name, "field_1")
            if all(field["id"] != live_id and field["name"] != live_name for field in layout["fields"]):
                layout["fields"].append(
                    self.normalize_field(
                        {
                            **DEFAULT_FIELD,
                            "id": live_id,
                            "name": live_name,
                            "active_row_id": field_map.get("active_row_id") or field_map.get("row_id") or "A",
                            "row_count": field_map.get("row_count") or 1,
                            "plants_per_row": field_map.get("plants_per_row") or 100,
                            "row_spacing_m": field_map.get("row_spacing_m") or 1.0,
                            "plant_spacing_m": field_map.get("plant_spacing_m") or 0.5,
                            "start_plant": field_map.get("start_plant") or 1,
                            "plant_step": field_map.get("plant_step") or 1,
                            "rows": field_map.get("rows") or [],
                        },
                        DEFAULT_FIELD,
                    )
                )
                layout["active_field_id"] = live_id

        seen_by_field = {}
        for row in decisions:
            name = row.get("field_id") or layout["fields"][0]["name"]
            seen_by_field[name] = max(seen_by_field.get(name, 0), int(row.get("plant_column") or row.get("plant_number") or row.get("plant_id") or 0))
        for field in layout["fields"]:
            seen = seen_by_field.get(field["name"], 0)
            if seen:
                field["plants_per_row"] = max(int(field["plants_per_row"]), seen)
                for row in field["rows"]:
                    row["plants_per_row"] = max(int(row.get("plants_per_row") or field["plants_per_row"]), seen)
        return layout

    def field_layout_payload(self, *, metadata: dict | None = None, decisions: list[dict] | None = None):
        return self.active_field_layout(self.farm_layout_payload(metadata=metadata, decisions=decisions))

    def active_field_layout(self, farm_layout: dict):
        fields = farm_layout.get("fields") or [self.normalize_field(DEFAULT_FIELD, DEFAULT_FIELD)]
        active_id = farm_layout.get("active_field_id") or fields[0]["id"]
        active = next((field for field in fields if field["id"] == active_id), fields[0])
        layout = {
            **active,
            "field_id": active["name"],
            "field_name": active["name"],
            "active_field_id": active["id"],
            "fields": fields,
        }
        layout["row_ids"] = [row["id"] for row in active["rows"]]
        layout["planned_total_plants"] = sum(int(row.get("plants_per_row") or active["plants_per_row"]) for row in active["rows"])
        return layout

    def normalize_farm_layout(self, payload: dict, defaults: dict):
        if not isinstance(payload, dict):
            raise ValueError("layout must be an object")
        fields_payload = payload.get("fields")
        if not isinstance(fields_payload, list) or not fields_payload:
            fields_payload = [payload]
        if len(fields_payload) > MAX_FIELDS:
            raise ValueError(f"at most {MAX_FIELDS} fields are allowed")
        fields = []
        used_ids = set()
        for index, raw_field in enumerate(fields_payload, start=1):
            field = self.normalize_field(raw_field if isinstance(raw_field, dict) else {}, DEFAULT_FIELD, index=index)
            base_id = field["id"]
            suffix = 2
            while field["id"] in used_ids:
                field["id"] = f"{base_id}_{suffix}"
                suffix += 1
            used_ids.add(field["id"])
            fields.append(field)
        active = clean_id(payload.get("active_field_id"), fields[0]["id"])
        if active not in used_ids:
            active = fields[0]["id"]
        return {
            "active_field_id": active,
            "fields": fields,
            "updated_at": payload.get("updated_at") or datetime.now().astimezone().isoformat(timespec="seconds"),
        }

    def normalize_field(self, payload: dict, defaults: dict, *, index: int = 1):
        default_name = clean_text(defaults.get("name") or defaults.get("field_id"), f"Field {index}")
        name = clean_text(payload.get("name") or payload.get("field_name") or payload.get("field_id"), default_name, limit=80)
        field_id = clean_id(payload.get("id") or payload.get("active_field_id") or name, f"field_{index}")
        row_count = clamp_int(payload.get("row_count"), int(defaults.get("row_count", 1)), 1, 100)
        plants_per_row = clamp_int(payload.get("plants_per_row") or payload.get("default_plants_per_row"), int(defaults.get("plants_per_row", 100)), 1, 1000)
        row_spacing_m = clamp_float(payload.get("row_spacing_m"), float(defaults.get("row_spacing_m", 1.0)), 0.0, 1000.0)
        plant_spacing_m = clamp_float(payload.get("plant_spacing_m"), float(defaults.get("plant_spacing_m", 0.5)), 0.0, 1000.0)
        start_plant = clamp_int(payload.get("start_plant"), int(defaults.get("start_plant", 1)), 1, 100000)
        plant_step = clamp_int(payload.get("plant_step"), int(defaults.get("plant_step", 1)), 1, 1000)
        generated_rows = row_ids(row_count)
        active_default = clean_text(defaults.get("active_row_id"), "A", limit=12).upper()
        active_row = clean_text(payload.get("active_row_id") or payload.get("row_id"), active_default, limit=12).upper()
        if active_row not in generated_rows:
            active_row = generated_rows[0]

        row_payloads = {}
        raw_rows = payload.get("rows") or []
        if not isinstance(raw_rows, list):
            raw_rows = []
        for raw_row in raw_rows[:MAX_ROWS_PER_FIELD]:
            if not isinstance(raw_row, dict):
                continue
            row_id = clean_text(raw_row.get("id") or raw_row.get("row_id"), "", limit=12).upper()
            if row_id:
                row_payloads[row_id] = raw_row

        rows = []
        y_m = 0.0
        for row_index, row_id in enumerate(generated_rows, start=1):
            raw_row = row_payloads.get(row_id, {})
            row_plants = clamp_int(raw_row.get("plants_per_row"), plants_per_row, 1, 1000)
            row_plant_spacing = clamp_float(raw_row.get("plant_spacing_m"), plant_spacing_m, 0.0, 1000.0)
            row_spacing = clamp_float(raw_row.get("row_spacing_m"), row_spacing_m, 0.0, 1000.0)
            rows.append(
                {
                    "id": row_id,
                    "row_id": row_id,
                    "row_index": row_index,
                    "plants_per_row": row_plants,
                    "plant_spacing_m": row_plant_spacing,
                    "row_spacing_m": row_spacing,
                    "y_m": round(y_m, 4),
                }
            )
            y_m += row_spacing
        return {
            "id": field_id,
            "name": name,
            "field_id": name,
            "active_row_id": active_row,
            "row_count": row_count,
            "plants_per_row": plants_per_row,
            "row_spacing_m": row_spacing_m,
            "plant_spacing_m": plant_spacing_m,
            "start_plant": start_plant,
            "plant_step": plant_step,
            "plant_cooldown_sec": clamp_float(payload.get("plant_cooldown_sec"), float(defaults.get("plant_cooldown_sec", 2.0)), 0.0, 120.0),
            "rows": rows,
        }

    def field_summaries(self, farm_layout: dict, decisions: list[dict]):
        fields = farm_layout.get("fields") or []
        rows = []
        default_field_name = fields[0]["name"] if fields else DEFAULT_FIELD["name"]
        for field in fields:
            field_decisions = [row for row in decisions if (row.get("field_id") or default_field_name) == field["name"]]
            latest_by_key = {}
            for row in field_decisions:
                key = row.get("plant_key") or f"{field['name']}|{row.get('row_id', 'A')}|{row.get('plant_number') or row.get('plant_id') or row.get('sequence')}"
                latest_by_key[key] = row
            latest_rows = list(latest_by_key.values())
            sick = [
                row
                for row in latest_rows
                if row.get("status") == "ok"
                and finite_float(row.get("confidence"), 0.0) >= CONFIDENCE_THRESHOLD
                and str(row.get("label", "")).lower().replace("_", " ") not in {"healthy", "uncertain"}
            ]
            uncertain = [
                row
                for row in latest_rows
                if row.get("status") == "uncertain"
                or str(row.get("label", "")).lower() == "uncertain"
                or finite_float(row.get("confidence"), 0.0) < CONFIDENCE_THRESHOLD
            ]
            rows.append(
                {
                    "field_id": field["id"],
                    "field_name": field["name"],
                    "total_detected": len(latest_rows),
                    "sick": len(sick),
                    "uncertain": len(uncertain),
                    "diseases": disease_location_summary(latest_rows),
                    "latest_decision": field_decisions[-1] if field_decisions else None,
                }
            )
        return rows

    def normalize_field_layout(self, payload: dict, defaults: dict):
        return self.normalize_field(payload, defaults)

    def update_farmer_config(self, layout: dict):
        config = read_json(FARMER_CONFIG, {})
        if not isinstance(config, dict):
            config = {}
        active = self.active_field_layout(layout)
        config.update(
            {
                "field_id": active["field_id"],
                "row_id": active["active_row_id"],
                "start_plant": active["start_plant"],
                "plant_step": active["plant_step"],
                "plant_cooldown_sec": active["plant_cooldown_sec"],
                "field_layout": layout,
            }
        )
        atomic_write_text(FARMER_CONFIG, json.dumps(finite_json(config), indent=2, allow_nan=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Serve the Agribot local field dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, choices=range(1, 65536), default=8080)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--max-workers", type=int, choices=range(1, 33), default=8)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    (data_root / "runs").mkdir(parents=True, exist_ok=True)
    server = BoundedThreadingHTTPServer((args.host, args.port), AgribotHandler, max_workers=args.max_workers)
    server.data_root = data_root
    server.quiet = args.quiet
    server.write_lock = threading.Lock()
    server.auth_token = os.environ.get("AGRIBOT_PLATFORM_TOKEN", "").strip()
    server.allowed_origins = {
        origin.strip().rstrip("/")
        for origin in os.environ.get("AGRIBOT_PLATFORM_CORS_ORIGINS", "").split(",")
        if origin.strip()
    }
    ip = now_ip()
    print("Agribot platform is running")
    print(f"  local : http://127.0.0.1:{args.port}")
    print(f"  phone : http://{ip}:{args.port}")
    print(f"  data  : {data_root}")
    print(f"  auth  : {'configured bearer token' if server.auth_token else 'same-origin/loopback writes (set AGRIBOT_PLATFORM_TOKEN to require bearer auth)'}")
    print(f"  workers: {args.max_workers}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Agribot platform.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
