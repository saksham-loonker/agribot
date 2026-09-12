import importlib.util
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PLATFORM = load_module("agribot_platform_run_platform", ROOT / "agribot_platform" / "run_platform.py")
FARMER = load_module("agribot_farmer_one_touch", ROOT / "44_farmer_one_touch.py")


class PlatformServer:
    def __init__(self, *, token: str = ""):
        self.tempdir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.tempdir.name)
        self.old_farmer_config = PLATFORM.FARMER_CONFIG
        PLATFORM.FARMER_CONFIG = self.data_root / "farmer_config.json"
        self.server = PLATFORM.BoundedThreadingHTTPServer(("127.0.0.1", 0), PLATFORM.AgribotHandler, max_workers=2)
        self.server.data_root = self.data_root
        self.server.quiet = True
        self.server.write_lock = threading.Lock()
        self.server.auth_token = token
        self.server.allowed_origins = set()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        PLATFORM.FARMER_CONFIG = self.old_farmer_config
        self.tempdir.cleanup()


def call(server: PlatformServer, path: str, *, method="GET", body=None, headers=None):
    request = urllib.request.Request(server.base_url + path, method=method, headers=headers or {})
    if body is not None:
        request.data = body
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status, response.read(), response.headers
    except urllib.error.HTTPError as error:
        return error.code, error.read(), error.headers


class PlatformSecurityTests(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "platform"):
            self.platform.close()

    def post_layout(self, platform, payload, *, token="", origin=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if origin:
            headers["Origin"] = origin
        return call(platform, "/api/farm-layout", method="POST", body=json.dumps(payload).encode(), headers=headers)

    def test_non_finite_json_is_rejected_without_writing(self):
        self.platform = PlatformServer()
        origin = self.platform.base_url
        status, _, _ = call(
            self.platform,
            "/api/farm-layout",
            method="POST",
            body=b'{"fields":[{"name":"Field 1","row_spacing_m":NaN}]}',
            headers={"Content-Type": "application/json", "Origin": origin},
        )
        self.assertEqual(status, 400)
        self.assertFalse((self.platform.data_root / "field_layout.json").exists())

    def test_numeric_string_nan_is_rejected(self):
        self.platform = PlatformServer()
        status, _, _ = self.post_layout(
            self.platform,
            {"fields": [{"name": "Field 1", "row_spacing_m": "NaN"}]},
            origin=self.platform.base_url,
        )
        self.assertEqual(status, 400)

    def test_cross_origin_write_is_rejected_without_token(self):
        self.platform = PlatformServer()
        status, _, _ = self.post_layout(
            self.platform,
            {"fields": [{"name": "Field 1"}]},
            origin="http://evil.example",
        )
        self.assertEqual(status, 403)
        self.assertFalse((self.platform.data_root / "field_layout.json").exists())

    def test_same_origin_write_keeps_local_offline_operation(self):
        self.platform = PlatformServer()
        status, body, _ = self.post_layout(
            self.platform,
            {"fields": [{"name": "Field 1", "row_count": 2}]},
            origin=self.platform.base_url,
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["farm_layout"]["fields"][0]["row_count"], 2)

    def test_configured_token_is_required_for_writes(self):
        self.platform = PlatformServer(token="test-local-token")
        origin = self.platform.base_url
        status, _, _ = self.post_layout(self.platform, {"fields": [{"name": "Field 1"}]}, origin=origin)
        self.assertEqual(status, 401)
        status, _, _ = self.post_layout(
            self.platform,
            {"fields": [{"name": "Field 1"}]},
            token="test-local-token",
            origin=origin,
        )
        self.assertEqual(status, 200)

    def test_request_body_and_run_id_are_bounded(self):
        self.platform = PlatformServer()
        origin = self.platform.base_url
        huge = b"{" + b'"fields":[' + b"{}" * 40_000 + b"]}"
        status, _, _ = call(
            self.platform,
            "/api/farm-layout",
            method="POST",
            body=huge,
            headers={"Content-Type": "application/json", "Origin": origin},
        )
        self.assertEqual(status, 413)
        status, _, _ = call(self.platform, "/api/runs/%2e%2e/decisions")
        self.assertEqual(status, 400)

    def test_csv_export_neutralizes_formula_cells(self):
        self.platform = PlatformServer()
        run_dir = self.platform.data_root / "runs" / "run_1"
        run_dir.mkdir(parents=True)
        (run_dir / "events.csv").write_text("label,notes\n=cmd|' /C calc'!,@user\n", encoding="utf-8")
        status, body, headers = call(self.platform, "/api/download/run_1.csv")
        self.assertEqual(status, 200)
        self.assertEqual(headers.get_content_type(), "text/csv")
        self.assertIn(b"'=cmd", body)
        self.assertIn(b"'@user", body)


class ArchivePathTests(unittest.TestCase):
    def test_archive_target_cannot_escape_repository(self):
        with self.assertRaises(ValueError):
            FARMER.safe_archive_target("../outside")
        with self.assertRaises(ValueError):
            FARMER.safe_archive_target(FARMER.ROOT)


if __name__ == "__main__":
    unittest.main()
