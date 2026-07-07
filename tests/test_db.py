"""Unit tests for the DB layer."""

import os
import tempfile
import unittest
from datetime import timedelta

from db import Database, utcnow


class DbTestCase(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db = Database(f"sqlite:///{self.path}")

    def tearDown(self):
        self.db.close()
        os.unlink(self.path)

    def test_create_and_fetch_post(self):
        post_id = self.db.create_post("affiliate", "B01", "board1", "posted", pin_id="pin1")
        self.assertIsInstance(post_id, int)
        posts = self.db.recent_posts()
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0]["pin_id"], "pin1")
        self.assertEqual(posts[0]["content_type"], "affiliate")

    def test_update_post(self):
        post_id = self.db.create_post("organic", "topic", "board1", "pending")
        self.db.update_post(post_id, status="posted", pin_id="pinX", pin_url="http://x")
        post = self.db.recent_posts()[0]
        self.assertEqual(post["status"], "posted")
        self.assertEqual(post["pin_id"], "pinX")

    def test_recent_affiliate_ratio(self):
        self.assertEqual(self.db.recent_affiliate_ratio(10), 0.0)
        self.db.create_post("affiliate", "a", "b", "posted")
        self.db.create_post("organic", "o", "b", "posted")
        self.db.create_post("affiliate", "a2", "b", "posted")
        # 2 of 3 affiliate.
        self.assertAlmostEqual(self.db.recent_affiliate_ratio(10), 2 / 3)

    def test_ratio_ignores_non_posted(self):
        self.db.create_post("affiliate", "a", "b", "skipped")
        self.db.create_post("organic", "o", "b", "posted")
        self.assertEqual(self.db.recent_affiliate_ratio(10), 0.0)

    def test_product_reuse_tracking(self):
        self.db.upsert_product("B01", "Widget", 0.05, "src")
        self.assertFalse(self.db.product_used_within("B01", 14))
        self.db.mark_product_used("B01")
        self.assertTrue(self.db.product_used_within("B01", 14))
        self.assertIn("B01", self.db.products_used_within(14))

    def test_product_used_outside_window(self):
        self.db.upsert_product("B02", "Widget2")
        self.db.mark_product_used("B02", when=utcnow() - timedelta(days=30))
        self.assertFalse(self.db.product_used_within("B02", 14))
        self.assertNotIn("B02", self.db.products_used_within(14))

    def test_topic_used_within(self):
        self.assertFalse(self.db.topic_used_within("declutter", 14))
        self.db.create_post("organic", "declutter", "b", "posted")
        self.assertTrue(self.db.topic_used_within("declutter", 14))

    def test_verifier_log(self):
        self.db.log_verifier("attempt1", False, ["missing disclosure"])
        self.db.log_verifier("attempt2", True, [])
        # No fetch API required; just ensure no exceptions and rows exist.
        with self.db._cursor() as cur:
            cur.execute("SELECT COUNT(*) AS c FROM verifier_log")
            self.assertEqual(cur.fetchone()["c"], 2)

    def test_alerts(self):
        self.db.add_alert("something broke")
        alerts = self.db.recent_alerts()
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["message"], "something broke")

    def test_api_usage_counter(self):
        self.assertEqual(self.db.api_usage_today("openrouter"), 0)
        self.assertEqual(self.db.increment_api_usage("openrouter"), 1)
        self.assertEqual(self.db.increment_api_usage("openrouter"), 2)
        self.assertEqual(self.db.api_usage_today("openrouter"), 2)
        self.assertEqual(self.db.api_usage_today("higgsfield"), 0)

    def test_last_successful_post_time(self):
        self.assertIsNone(self.db.last_successful_post_time())
        self.db.create_post("organic", "o", "b", "posted")
        self.assertIsNotNone(self.db.last_successful_post_time())

    def test_performance_record(self):
        post_id = self.db.create_post("affiliate", "a", "b", "posted", pin_id="pin1")
        self.db.record_performance(post_id, saves=5, clicks=2)
        self.assertEqual(len(self.db.posts_with_pins()), 1)


if __name__ == "__main__":
    unittest.main()
