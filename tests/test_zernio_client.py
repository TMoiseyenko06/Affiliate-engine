"""Unit tests for the Zernio client (Pinterest posting via a third-party
scheduler) against the documented request/response contract."""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from agents.clients import ApiError, ZernioClient  # noqa: E402
from config import CONFIG  # noqa: E402


def _with_zernio_config(api_key="testkey", account_id="acct123"):
    """Config singletons are process-wide and may already be instantiated by
    another test module without Zernio env vars set, so patch attributes
    directly rather than relying on os.environ (which only affects values
    read at CONFIG construction time)."""
    return patch.multiple(CONFIG, zernio_api_key=api_key, zernio_pinterest_account_id=account_id)


class FakeResp:
    def __init__(self, js=None):
        self._js = js

    def raise_for_status(self):
        pass

    def json(self):
        return self._js


def presign_response():
    return FakeResp({
        "uploadUrl": "https://bucket.r2.cloudflarestorage.com/temp/abc",
        "publicUrl": "https://media.zernio.com/temp/abc",
        "key": "temp/abc",
        "expiresIn": 3600,
    })


class ZernioUploadTests(unittest.TestCase):
    def setUp(self):
        patcher = _with_zernio_config()
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_upload_media_follows_presign_then_put(self):
        calls = []

        def fake_post(url, headers=None, json=None, timeout=None):
            calls.append(("POST", url, json))
            return presign_response()

        def fake_put(url, data=None, headers=None, timeout=None):
            calls.append(("PUT", url, headers.get("Content-Type")))
            return FakeResp()

        with patch("requests.post", side_effect=fake_post), patch("requests.put", side_effect=fake_put):
            url = ZernioClient().upload_media(b"BYTES", content_type="image/jpeg")

        self.assertEqual(url, "https://media.zernio.com/temp/abc")
        self.assertEqual(calls[0], ("POST", "https://zernio.com/api/v1/media/presign", {"filename": "pin.jpg", "contentType": "image/jpeg"}))
        self.assertEqual(calls[1], ("PUT", "https://bucket.r2.cloudflarestorage.com/temp/abc", "image/jpeg"))

    def test_upload_media_missing_key_raises(self):
        with patch("requests.post", return_value=FakeResp({"key": "x"})):
            with self.assertRaises(ApiError):
                ZernioClient().upload_media(b"BYTES")


class ZernioCreatePostTests(unittest.TestCase):
    def setUp(self):
        patcher = _with_zernio_config()
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_success_status_returns_response(self):
        def fake_post(url, headers=None, json=None, timeout=None):
            self.assertEqual(json["mediaItems"], [{"type": "image", "url": "https://img"}])
            self.assertEqual(json["platforms"][0]["accountId"], "acct123")
            self.assertEqual(json["platforms"][0]["platformSpecificData"]["boardId"], "board1")
            self.assertTrue(json["publishNow"])
            return FakeResp({
                "post": {
                    "_id": "p1",
                    "platforms": [{"platform": "pinterest", "status": "success", "postUrl": "https://pin/1"}],
                }
            })

        with patch("requests.post", side_effect=fake_post):
            result = ZernioClient().create_pinterest_post("t", "d", "https://img", "board1")

        self.assertEqual(ZernioClient.extract_post_id(result), "p1")
        self.assertEqual(ZernioClient.extract_pin_url(result), "https://pin/1")

    def test_failed_status_raises_apierror(self):
        def fake_post(url, headers=None, json=None, timeout=None):
            return FakeResp({
                "post": {
                    "_id": "p2",
                    "platforms": [{"platform": "pinterest", "status": "failed", "error": "Invalid URL or request data."}],
                }
            })

        with patch("requests.post", side_effect=fake_post):
            with self.assertRaises(ApiError) as ctx:
                ZernioClient().create_pinterest_post("t", "d", "https://img", "board1")
        self.assertIn("Invalid URL", str(ctx.exception))

    def test_unrecognized_status_does_not_raise(self):
        def fake_post(url, headers=None, json=None, timeout=None):
            return FakeResp({"post": {"_id": "p3", "platforms": [{"platform": "pinterest", "status": "processing"}]}})

        with patch("requests.post", side_effect=fake_post):
            result = ZernioClient().create_pinterest_post("t", "d", "https://img", "board1")
        self.assertEqual(ZernioClient.extract_post_id(result), "p3")

    def test_link_included_when_provided(self):
        def fake_post(url, headers=None, json=None, timeout=None):
            self.assertEqual(json["platforms"][0]["platformSpecificData"]["link"], "https://amazon.com/dp/X?tag=t-20")
            return FakeResp({"post": {"_id": "p4", "platforms": [{"platform": "pinterest", "status": "success"}]}})

        with patch("requests.post", side_effect=fake_post):
            ZernioClient().create_pinterest_post(
                "t", "d", "https://img", "board1", link="https://amazon.com/dp/X?tag=t-20"
            )

    def test_missing_account_id_raises(self):
        with patch.object(CONFIG, "zernio_pinterest_account_id", None):
            with self.assertRaises(ApiError):
                ZernioClient().create_pinterest_post("t", "d", "https://img", "board1")


if __name__ == "__main__":
    unittest.main()
