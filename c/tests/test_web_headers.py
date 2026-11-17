"""The headers sent with what a browser may show, and the cross-site check (web_headers.py)."""
import sys
import unittest
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(C_DIR))

import web_headers  # noqa: E402

ALLOWED = ("http://127.0.0.1:8000", "http://localhost:5173")


class CrossSiteTest(unittest.TestCase):
    def refused(self, path="/v1/chat/completions", site="cross-site", origin="https://evil.example", allowed=ALLOWED):
        return web_headers.cross_site_refused(path, site, origin, allowed)

    def test_a_page_on_another_site_is_refused(self):
        self.assertTrue(self.refused())
        self.assertTrue(self.refused(origin=None))                 # a no-cors request carries no Origin at all

    def test_what_sends_no_fetch_metadata_is_untouched(self):
        self.assertFalse(self.refused(site=None))                  # curl, the SDKs, the editor extension
        self.assertFalse(self.refused(site=""))

    def test_the_dashboard_and_its_neighbours_are_untouched(self):
        for site in ("same-origin", "same-site", "none"):
            self.assertFalse(self.refused(site=site), site)

    def test_an_allowed_origin_gets_through(self):
        self.assertFalse(self.refused(origin="http://localhost:5173"))
        self.assertFalse(self.refused(allowed=("*",)))

    def test_only_the_api_is_checked(self):
        for path in ("/", "/assets/index.js", "/health"):
            self.assertFalse(self.refused(path=path), path)        # a link from another site may open the dashboard


class PolicyTest(unittest.TestCase):
    def test_the_policy_keeps_scripts_to_our_own_files_and_refuses_framing(self):
        policy = dict(part.strip().split(" ", 1) for part in web_headers.CSP.split(";"))
        self.assertEqual(policy["script-src"], "'self'")
        self.assertEqual(policy["object-src"], "'none'")
        self.assertEqual(policy["frame-ancestors"], "'none'")
        self.assertEqual(policy["base-uri"], "'none'")
        self.assertNotIn("unsafe-eval", web_headers.CSP)
        self.assertNotIn("script-src 'self' 'unsafe-inline'", web_headers.CSP)

    def test_the_dashboard_can_show_what_it_makes(self):
        policy = dict(part.strip().split(" ", 1) for part in web_headers.CSP.split(";"))
        for kind in ("img-src", "media-src"):
            self.assertIn("blob:", policy[kind])
            self.assertIn("data:", policy[kind])
        self.assertIn("http:", policy["connect-src"])               # the Connection page can point it at another server


if __name__ == "__main__":
    unittest.main()
