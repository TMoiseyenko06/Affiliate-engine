"""Unit tests for the deterministic verifier checks."""

import os
import tempfile
import unittest

from config import (
    CONFIG,
    PINTEREST_DESCRIPTION_MAX,
    PINTEREST_TITLE_MAX,
    PIN_IMAGE_HEIGHT,
    PIN_IMAGE_WIDTH,
)
from db import Database
from agents.verifier_agent import run_deterministic_checks
from agents.copywriter_agent import build_associates_link


def good_image_meta(title_drawn=True):
    return {
        "width": PIN_IMAGE_WIDTH,
        "height": PIN_IMAGE_HEIGHT,
        "aspect_ratio": PIN_IMAGE_WIDTH / PIN_IMAGE_HEIGHT,
        "size_bytes": 500_000,
        "aspect_ok": True,
        "size_ok": True,
        "title_drawn": title_drawn,
    }


def good_affiliate_copy():
    link = build_associates_link(
        "https://www.amazon.com/dp/B01", CONFIG.amazon_associates_tag
    )
    return {
        "title": "Best Storage Bins for a Tidy Home",
        "description": (
            "Keep everything neat and tidy with these bins. "
            + CONFIG.affiliate_disclosure_text
        ),
        "keywords": ["storage", "organization"],
        "disclosure_text_or_null": CONFIG.affiliate_disclosure_text,
        "link": link,
        "link_or_null": link,
    }


class VerifierAffiliateTests(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db = Database(f"sqlite:///{self.path}")
        self.subject = {"product_id": "B01", "title": "Storage Bins", "category": "home"}

    def tearDown(self):
        self.db.close()
        os.unlink(self.path)

    def test_valid_affiliate_passes(self):
        failures = run_deterministic_checks(
            "affiliate", good_affiliate_copy(), good_image_meta(), self.subject, self.db
        )
        self.assertEqual(failures, [], failures)

    def test_missing_disclosure_fails(self):
        copy = good_affiliate_copy()
        copy["description"] = "Keep everything neat and tidy with these bins."
        failures = run_deterministic_checks(
            "affiliate", copy, good_image_meta(), self.subject, self.db
        )
        self.assertTrue(any("disclosure" in f.lower() for f in failures))

    def test_wrong_tag_fails(self):
        copy = good_affiliate_copy()
        copy["link"] = "https://www.amazon.com/dp/B01?tag=wrong-99"
        copy["link_or_null"] = copy["link"]
        failures = run_deterministic_checks(
            "affiliate", copy, good_image_meta(), self.subject, self.db
        )
        self.assertTrue(any("tag" in f.lower() for f in failures))

    def test_missing_tag_fails(self):
        copy = good_affiliate_copy()
        copy["link"] = "https://www.amazon.com/dp/B01"
        copy["link_or_null"] = copy["link"]
        failures = run_deterministic_checks(
            "affiliate", copy, good_image_meta(), self.subject, self.db
        )
        self.assertTrue(any("tag" in f.lower() for f in failures))

    def test_cloaked_link_fails(self):
        copy = good_affiliate_copy()
        copy["link"] = f"https://bit.ly/xyz?tag={CONFIG.amazon_associates_tag}"
        copy["link_or_null"] = copy["link"]
        failures = run_deterministic_checks(
            "affiliate", copy, good_image_meta(), self.subject, self.db
        )
        self.assertTrue(any("cloaking" in f.lower() or "amazon" in f.lower() for f in failures))

    def test_missing_link_fails(self):
        copy = good_affiliate_copy()
        copy["link"] = None
        copy["link_or_null"] = None
        failures = run_deterministic_checks(
            "affiliate", copy, good_image_meta(), self.subject, self.db
        )
        self.assertTrue(any("link" in f.lower() for f in failures))

    def test_wrong_dimensions_fail(self):
        meta = good_image_meta()
        meta["width"] = 800
        meta["aspect_ok"] = False
        failures = run_deterministic_checks(
            "affiliate", good_affiliate_copy(), meta, self.subject, self.db
        )
        self.assertTrue(any("dimension" in f.lower() or "aspect" in f.lower() for f in failures))

    def test_oversized_image_fails(self):
        meta = good_image_meta()
        meta["size_bytes"] = 50 * 1024 * 1024
        failures = run_deterministic_checks(
            "affiliate", good_affiliate_copy(), meta, self.subject, self.db
        )
        self.assertTrue(any("size" in f.lower() for f in failures))

    def test_title_too_long_fails(self):
        copy = good_affiliate_copy()
        copy["title"] = "x" * (PINTEREST_TITLE_MAX + 1)
        failures = run_deterministic_checks(
            "affiliate", copy, good_image_meta(), self.subject, self.db
        )
        self.assertTrue(any("title" in f.lower() for f in failures))

    def test_description_too_long_fails(self):
        copy = good_affiliate_copy()
        copy["description"] = "x" * (PINTEREST_DESCRIPTION_MAX + 1) + CONFIG.affiliate_disclosure_text
        failures = run_deterministic_checks(
            "affiliate", copy, good_image_meta(), self.subject, self.db
        )
        self.assertTrue(any("description" in f.lower() for f in failures))

    def test_duplicate_product_fails(self):
        # Simulate the product already being posted within the window.
        self.db.create_post("affiliate", "B01", "board", "posted")
        failures = run_deterministic_checks(
            "affiliate", good_affiliate_copy(), good_image_meta(), self.subject, self.db
        )
        self.assertTrue(any("already posted" in f.lower() for f in failures))


class VerifierAffiliateImageOnlyTests(unittest.TestCase):
    """affiliate_image_only shares all affiliate compliance rules (disclosure,
    link, tag, dedup) but must have NO title drawn on the image."""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db = Database(f"sqlite:///{self.path}")
        self.subject = {"product_id": "B01", "title": "Storage Bins", "category": "home"}

    def tearDown(self):
        self.db.close()
        os.unlink(self.path)

    def test_valid_image_only_passes(self):
        failures = run_deterministic_checks(
            "affiliate_image_only", good_affiliate_copy(), good_image_meta(title_drawn=False),
            self.subject, self.db,
        )
        self.assertEqual(failures, [], failures)

    def test_image_only_still_requires_disclosure(self):
        copy = good_affiliate_copy()
        copy["description"] = "Keep everything neat and tidy with these bins."
        failures = run_deterministic_checks(
            "affiliate_image_only", copy, good_image_meta(title_drawn=False), self.subject, self.db
        )
        self.assertTrue(any("disclosure" in f.lower() for f in failures))

    def test_image_only_still_requires_correct_link(self):
        copy = good_affiliate_copy()
        copy["link"] = "https://bit.ly/xyz?tag=example-20"
        copy["link_or_null"] = copy["link"]
        failures = run_deterministic_checks(
            "affiliate_image_only", copy, good_image_meta(title_drawn=False), self.subject, self.db
        )
        self.assertTrue(any("amazon" in f.lower() or "cloaking" in f.lower() for f in failures))

    def test_image_only_dedup_still_applies(self):
        self.db.create_post("affiliate_image_only", "B01", "board", "posted")
        failures = run_deterministic_checks(
            "affiliate_image_only", good_affiliate_copy(), good_image_meta(title_drawn=False),
            self.subject, self.db,
        )
        self.assertTrue(any("already posted" in f.lower() for f in failures))


class TitleOverlayConsistencyTests(unittest.TestCase):
    """The compositor's title-drawing intent must match content_type."""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db = Database(f"sqlite:///{self.path}")
        self.subject = {"product_id": "B01", "title": "Storage Bins", "category": "home"}

    def tearDown(self):
        self.db.close()
        os.unlink(self.path)

    def test_affiliate_with_title_not_drawn_fails(self):
        failures = run_deterministic_checks(
            "affiliate", good_affiliate_copy(), good_image_meta(title_drawn=False), self.subject, self.db
        )
        self.assertTrue(any("title overlay" in f.lower() for f in failures))

    def test_image_only_with_title_drawn_fails(self):
        failures = run_deterministic_checks(
            "affiliate_image_only", good_affiliate_copy(), good_image_meta(title_drawn=True),
            self.subject, self.db,
        )
        self.assertTrue(any("title overlay" in f.lower() for f in failures))


class VerifierOrganicTests(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db = Database(f"sqlite:///{self.path}")
        self.subject = {"niche": "home", "topic": "declutter tips"}

    def tearDown(self):
        self.db.close()
        os.unlink(self.path)

    def good_organic_copy(self):
        return {
            "title": "5 Calming Declutter Habits for a Peaceful Home",
            "description": "Small daily habits create a serene, tidy space over time.",
            "keywords": ["declutter", "calm"],
            "disclosure_text_or_null": None,
            "link_or_null": None,
            "link": None,
        }

    def test_valid_organic_passes(self):
        failures = run_deterministic_checks(
            "organic", self.good_organic_copy(), good_image_meta(), self.subject, self.db
        )
        self.assertEqual(failures, [], failures)

    def test_organic_with_link_fails(self):
        copy = self.good_organic_copy()
        copy["link_or_null"] = "https://www.amazon.com/dp/B01?tag=example-20"
        failures = run_deterministic_checks(
            "organic", copy, good_image_meta(), self.subject, self.db
        )
        self.assertTrue(any("link" in f.lower() for f in failures))

    def test_organic_with_cta_fails(self):
        copy = self.good_organic_copy()
        copy["description"] = "Shop now for the best declutter tools!"
        failures = run_deterministic_checks(
            "organic", copy, good_image_meta(), self.subject, self.db
        )
        self.assertTrue(any("cta" in f.lower() or "sales" in f.lower() for f in failures))

    def test_organic_with_disclosure_fails(self):
        copy = self.good_organic_copy()
        copy["disclosure_text_or_null"] = CONFIG.affiliate_disclosure_text
        failures = run_deterministic_checks(
            "organic", copy, good_image_meta(), self.subject, self.db
        )
        self.assertTrue(any("disclosure" in f.lower() for f in failures))


class LinkBuilderTests(unittest.TestCase):
    def test_tag_added(self):
        link = build_associates_link("https://www.amazon.com/dp/B01", "mytag-20")
        self.assertIn("tag=mytag-20", link)

    def test_tag_replaced(self):
        link = build_associates_link("https://www.amazon.com/dp/B01?tag=old-99", "new-20")
        self.assertIn("tag=new-20", link)
        self.assertNotIn("old-99", link)

    def test_existing_params_preserved(self):
        link = build_associates_link("https://www.amazon.com/dp/B01?ref=abc", "new-20")
        self.assertIn("ref=abc", link)
        self.assertIn("tag=new-20", link)


if __name__ == "__main__":
    unittest.main()
