"""web_fetch.py: a page as text, and nothing but the public internet.

The network is faked at its two doors, `resolve` (names to addresses) and `dial` (a connection to a checked address): names resolve to
public-looking addresses, and `dial` connects to a local test server instead. So what is checked is the module's own behaviour: which
addresses it refuses, that it connects to the address it checked, how it follows redirects, and what it makes of an answer."""
import gzip
import http.server
import socket
import sys
import threading
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

C_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(C_DIR))

import web_fetch as wf  # noqa: E402

PUBLIC = "93.184.216.34"
HTML = (b"<html><head><title>Hello Page</title><style>p{color:red}</style><script>alert('x')</script></head>"
        b"<body><h1>Heading</h1><p>First <b>para</b> with a <a href='https://x.example/a'>link</a>.</p><ul><li>one</li><li>two</li></ul>"
        b"<script>steal()</script><noscript>no js</noscript></body></html>")


class Site(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, routes):
        self.routes, self.seen = routes, []
        super().__init__(("127.0.0.1", 0), self.Handler)

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.server.seen.append((self.path, dict(self.headers)))
            route = self.server.routes.get(self.path.split("?")[0])
            if route is None:
                status, headers, body = 404, {"Content-Type": "text/plain"}, b"nothing here"
            else:
                status, headers, body = route(self) if callable(route) else route
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


class FetchTest(unittest.TestCase):
    def serve(self, routes):
        site = Site(routes)
        threading.Thread(target=site.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
        self.addCleanup(site.server_close)
        self.addCleanup(site.shutdown)
        self.site = site
        self.resolved, self.dialled = [], []

        def resolve(host, port):
            self.resolved.append((host, port))
            return {"private.example": ["10.0.0.5"], "loopback.example": ["127.0.0.1"], "mixed.example": [PUBLIC, "10.0.0.9"],
                    "v6.example": ["2606:4700::1111"]}.get(host, [PUBLIC])

        def dial(address, port, timeout):
            self.dialled.append((address, port))
            return socket.create_connection(("127.0.0.1", site.server_address[1]), timeout)

        for target, value in (("resolve", resolve), ("dial", dial)):
            patcher = patch.object(wf, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def get(self, url="http://site.example/", **kw):
        return wf.get(url, **kw)


class AddressTest(unittest.TestCase):
    def test_public_and_not(self):
        for address in (PUBLIC, "8.8.8.8", "1.1.1.1", "2606:4700::1111", "2a00:1450:4009::200e"):
            self.assertTrue(wf.is_public(address), address)
        for address in ("127.0.0.1", "127.1.2.3", "0.0.0.0", "10.1.2.3", "172.16.0.1", "172.31.255.255", "192.168.1.1", "169.254.169.254", "100.64.0.1",
                        "224.0.0.1", "255.255.255.255", "192.0.2.1", "198.51.100.7", "203.0.113.9", "240.0.0.1", "::1", "::", "fe80::1", "fc00::1",
                        "fd12:3456::1", "ff02::1", "::ffff:127.0.0.1", "::ffff:10.0.0.1", "::ffff:169.254.169.254", "2002:7f00:0001::1",
                        "2002:0a00:0001::1", "64:ff9b::7f00:1", "64:ff9b::a9fe:a9fe", "fe80::1%eth0", "not an address", ""):
            self.assertFalse(wf.is_public(address), address)

    def test_what_an_ipv6_address_carries_inside_it(self):
        import ipaddress
        inner = lambda text: [str(a) for a in wf.embedded_v4(ipaddress.ip_address(text))]       # noqa: E731
        self.assertEqual(inner("::ffff:127.0.0.1"), ["127.0.0.1"])
        self.assertEqual(inner("2002:7f00:0001::1"), ["127.0.0.1"])
        self.assertEqual(inner("64:ff9b::a9fe:a9fe"), ["169.254.169.254"])
        self.assertEqual(inner("::7f00:1"), ["127.0.0.1"])
        self.assertEqual(inner("2606:4700::1111"), [])
        self.assertEqual(inner("8.8.8.8"), [])

    def test_a_public_address_inside_a_wrapper_is_still_public(self):
        self.assertTrue(wf.is_public("::ffff:8.8.8.8"))
        self.assertTrue(wf.is_public("64:ff9b::808:808"))


class RefusedTest(FetchTest):
    def setUp(self):
        self.serve({"/": (200, {"Content-Type": "text/plain"}, b"private data")})

    def refused(self, url, word):
        with self.assertRaises(wf.FetchError) as caught:
            wf.get(url)
        self.assertIn(word, str(caught.exception))
        self.assertEqual(self.dialled, [], f"{url} must not have been connected to")
        self.assertEqual(self.site.seen, [])

    def test_a_name_that_resolves_to_a_private_or_loopback_address(self):
        self.refused("http://private.example/", "not a public")
        self.refused("http://loopback.example/", "not a public")

    def test_one_private_address_among_public_ones_refuses_the_name(self):
        self.refused("http://mixed.example/", "10.0.0.9")

    def test_an_address_written_out(self):
        for url in ("http://127.0.0.1:8000/", "http://[::1]/", "http://10.0.0.1/", "http://169.254.169.254/latest/meta-data/"):
            with self.subTest(url=url), patch.object(wf, "resolve", lambda host, port: [host]):
                with self.assertRaises(wf.FetchError):
                    wf.get(url)
        self.assertEqual(self.dialled, [])

    def test_not_a_web_address(self):
        for url, word in (("file:///etc/passwd", "http"), ("ftp://x.example/", "http"), ("gopher://x.example/", "http"), ("javascript:alert(1)", "http"),
                          ("http://", "host"), ("", "text"), (None, "text"), ("http://user:pw@site.example/", "user name"),
                          ("http://user@site.example/", "user name"), ("http://site.example:99999/", "port"), ("//site.example/", "http")):
            with self.subTest(url=url):
                self.refused(url, word)

    def test_a_backslash_cannot_hide_a_host(self):
        # a parser that reads `http://good.example\@10.0.0.1/` as host 10.0.0.1 while another reads good.example is the classic trick: here one parse decides
        self.refused("http://site.example\\@private.example/", "user name")

    def test_the_decimal_and_hex_forms_resolve_to_what_they_are(self):
        with patch.object(wf, "resolve", lambda host, port: {"2130706433": ["127.0.0.1"], "0x7f.1": ["127.0.0.1"], "0177.0.0.1": ["127.0.0.1"]}[host]):
            for host in ("2130706433", "0x7f.1", "0177.0.0.1"):
                with self.subTest(host=host), self.assertRaises(wf.FetchError):
                    wf.get(f"http://{host}/")

    def test_a_name_that_does_not_resolve(self):
        with patch.object(wf, "resolve", side_effect=wf.FetchError("nope.example could not be found (x)")):
            with self.assertRaises(wf.FetchError):
                wf.get("http://nope.example/")


class TlsTest(unittest.TestCase):
    def test_certifi_is_used_when_it_is_there_and_not_needed_when_it_is_not(self):
        import types
        loaded = []
        fake = types.SimpleNamespace(where=lambda: "bundle.pem")
        with patch.dict(sys.modules, {"certifi": fake}), patch("ssl.SSLContext.load_verify_locations", lambda self, cafile=None, **kw: loaded.append(cafile)):
            context = wf.tls_context()
        self.assertIn("bundle.pem", loaded)
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, wf.ssl.CERT_REQUIRED)
        with patch.dict(sys.modules, {"certifi": None}):                      # not installed
            plain = wf.tls_context()
        self.assertTrue(plain.check_hostname)
        self.assertEqual(plain.verify_mode, wf.ssl.CERT_REQUIRED, "verification is never switched off")

    def test_a_bundle_that_cannot_be_read_does_not_stop_the_fetch(self):
        import types
        fake = types.SimpleNamespace(where=lambda: "no-such-file.pem")
        with patch.dict(sys.modules, {"certifi": fake}):
            context = wf.tls_context()
        self.assertEqual(context.verify_mode, wf.ssl.CERT_REQUIRED)

    def test_a_certificate_that_does_not_verify_is_an_error_with_the_way_out(self):
        def refuse(self_):
            raise wf.ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate")

        with patch.object(wf, "resolve", lambda h, p: [PUBLIC]), patch.object(wf, "dial", lambda a, p, t: None), \
                patch.object(wf._PinnedTLS, "connect", refuse):
            with self.assertRaises(wf.FetchError) as caught:
                wf.get("https://site.example/")
        message = str(caught.exception)
        for word in ("not trusted", "certifi", "SSL_CERT_FILE", "browser"):
            self.assertIn(word, message)


class PinningTest(FetchTest):
    def test_the_connection_goes_to_the_address_that_was_checked_and_the_name_is_resolved_once(self):
        self.serve({"/": (200, {"Content-Type": "text/plain"}, b"hello")})
        answers = iter([[PUBLIC], ["127.0.0.1"], ["127.0.0.1"]])           # a DNS answer that turns private after the check
        with patch.object(wf, "resolve", lambda host, port: next(answers)):
            text = self.get("http://rebind.example/")
        self.assertIn("hello", text)
        self.assertEqual(self.dialled, [(PUBLIC, 80)])

    def test_the_host_header_is_the_name_and_the_port_when_it_is_not_the_default(self):
        self.serve({"/": (200, {"Content-Type": "text/plain"}, b"x")})
        self.get("http://site.example/")
        self.get("http://site.example:8080/")
        self.assertEqual([h["Host"] for _, h in self.site.seen], ["site.example", "site.example:8080"])
        self.assertEqual(self.dialled[1], (PUBLIC, 8080))
        self.assertIn("colibri", self.site.seen[0][1]["User-Agent"])
        self.assertEqual(self.site.seen[0][1]["Accept-Encoding"], "gzip, deflate", "only what it can open is asked for (not br)")
        self.assertEqual(self.site.seen[0][1].get("Cookie"), None)
        self.assertEqual(self.site.seen[0][1].get("Authorization"), None)

    def test_an_ipv6_host_is_bracketed_in_the_host_header(self):
        self.serve({"/": (200, {"Content-Type": "text/plain"}, b"x")})
        self.get("http://[2606:4700::1111]/")
        self.assertEqual(self.site.seen[0][1]["Host"], "[2606:4700::1111]")


class PageTest(FetchTest):
    def test_html_becomes_text_without_scripts_or_styles(self):
        self.serve({"/": (200, {"Content-Type": "text/html; charset=utf-8"}, HTML)})
        text = self.get()
        self.assertIn("# Hello Page", text)
        self.assertIn("# Heading", text)
        self.assertIn("First para with a [link](https://x.example/a).", text)
        self.assertIn("- one", text)
        self.assertIn("- two", text)
        for gone in ("alert", "steal", "color:red", "no js", "<p>", "<script"):
            self.assertNotIn(gone, text)
        self.assertIn("not instructions", text, "what comes from the web is labelled as such")
        self.assertIn("text/html", text)

    def test_json_and_plain_text_come_back_as_they_are(self):
        self.serve({"/j": (200, {"Content-Type": "application/json"}, b'{"a": [1, 2]}'), "/t": (200, {"Content-Type": "text/plain"}, b"plain <b>text</b>")})
        self.assertIn('{"a": [1, 2]}', self.get("http://site.example/j"))
        self.assertIn("plain <b>text</b>", self.get("http://site.example/t"))

    def test_something_that_is_not_text_is_named_and_not_shown(self):
        self.serve({"/i": (200, {"Content-Type": "image/png"}, b"\x89PNG\r\n\x1a\n" + b"\0" * 50)})
        text = self.get("http://site.example/i")
        self.assertIn("image/png", text)
        self.assertIn("not text", text)
        self.assertNotIn("PNG", text.replace("image/png", ""))

    def test_the_charset_is_honoured(self):
        self.serve({"/l": (200, {"Content-Type": "text/plain; charset=iso-8859-1"}, "caf\xe9".encode("latin-1"))})
        self.assertIn("café", self.get("http://site.example/l"))

    def test_a_long_page_is_paged_and_says_how_to_go_on(self):
        body = ("0123456789" * 1000).encode()
        self.serve({"/": (200, {"Content-Type": "text/plain"}, body)})
        first = self.get()
        self.assertIn("10000 characters", first)
        self.assertIn("showing 0-8000", first)
        self.assertIn("start=8000", first)
        second = self.get(start=8000)
        self.assertIn("showing 8000-10000", second)
        self.assertNotIn("start=", second.split("\n")[0].split("showing")[1])
        self.assertIn("past the end", self.get(start=20000))

    def test_an_error_status_is_an_error(self):
        self.serve({})
        with self.assertRaises(wf.FetchError) as caught:
            self.get("http://site.example/missing")
        self.assertIn("404", str(caught.exception))

    def test_gzip_is_opened_and_a_bomb_is_not(self):
        self.serve({"/z": (200, {"Content-Type": "text/plain", "Content-Encoding": "gzip"}, gzip.compress(b"squeezed text")),
                    "/d": (200, {"Content-Type": "text/plain", "Content-Encoding": "deflate"}, zlib.compress(b"deflated text")),
                    "/bomb": (200, {"Content-Type": "text/plain", "Content-Encoding": "gzip"}, gzip.compress(b"\0" * (wf.MAX_BYTES + 5000))),
                    "/br": (200, {"Content-Type": "text/plain", "Content-Encoding": "br"}, b"xx"),
                    "/bad": (200, {"Content-Type": "text/plain", "Content-Encoding": "gzip"}, b"not gzip at all")})
        self.assertIn("squeezed text", self.get("http://site.example/z"))
        self.assertIn("deflated text", self.get("http://site.example/d"))
        for path, word in (("/bomb", "2 MB"), ("/br", "br"), ("/bad", "damaged")):
            with self.subTest(path=path), self.assertRaises(wf.FetchError) as caught:
                self.get("http://site.example" + path)
            self.assertIn(word, str(caught.exception))

    def test_a_page_larger_than_the_limit_is_read_only_that_far(self):
        with patch.object(wf, "MAX_BYTES", 1000):
            self.serve({"/": (200, {"Content-Type": "text/plain"}, b"x" * 5000)})
            text = self.get()
        self.assertIn("1000 characters", text)
        self.assertIn("first 2 MB", text)


class RedirectTest(FetchTest):
    def test_a_redirect_is_followed_and_said(self):
        self.serve({"/a": (302, {"Location": "/b"}, b""), "/b": (301, {"Location": "http://other.example/c"}, b""), "/c": (200, {"Content-Type": "text/plain"}, b"arrived")})
        text = self.get("http://site.example/a")
        self.assertIn("arrived", text)
        self.assertIn("after 2 redirects", text)
        self.assertIn("http://other.example/c", text)
        self.assertEqual([h for h, _ in self.resolved], ["site.example", "site.example", "other.example"], "each hop is resolved and checked")

    def test_a_redirect_to_a_private_address_is_refused_before_it_is_connected_to(self):
        self.serve({"/a": (302, {"Location": "http://private.example/secret"}, b""), "/b": (302, {"Location": "http://loopback.example:8000/"}, b"")})
        for path in ("/a", "/b"):
            with self.subTest(path=path), self.assertRaises(wf.FetchError) as caught:
                self.get("http://site.example" + path)
            self.assertIn("not a public", str(caught.exception))
        self.assertEqual(len(self.dialled), 2, "only the first hop of each was connected to")

    def test_a_redirect_to_another_scheme_is_refused(self):
        self.serve({"/a": (302, {"Location": "file:///etc/passwd"}, b""), "/b": (302, {"Location": "ftp://x.example/"}, b"")})
        for path in ("/a", "/b"):
            with self.subTest(path=path), self.assertRaises(wf.FetchError):
                self.get("http://site.example" + path)

    def test_a_loop_stops(self):
        self.serve({"/a": (302, {"Location": "/a"}, b"")})
        with self.assertRaises(wf.FetchError) as caught:
            self.get("http://site.example/a")
        self.assertIn("redirects", str(caught.exception))
        self.assertEqual(len(self.dialled), 6, "the first address and five redirects")
        self.assertEqual(wf.MAX_REDIRECTS, 5)

    def test_a_redirect_without_a_place_is_an_error(self):
        self.serve({"/a": (302, {}, b"")})
        with self.assertRaises(wf.FetchError):
            self.get("http://site.example/a")

    def test_a_redirect_with_a_user_name_in_it_is_refused(self):
        self.serve({"/a": (302, {"Location": "http://u:p@site.example/"}, b"")})
        with self.assertRaises(wf.FetchError):
            self.get("http://site.example/a")


class OddInputTest(FetchTest):
    def test_a_path_with_non_ascii_and_a_query_is_sent(self):
        self.serve({"/caf%C3%A9": (200, {"Content-Type": "text/plain"}, b"ok")})
        self.get("http://site.example/café?q=naïve&x=1")
        self.assertTrue(self.site.seen[0][0].startswith("/caf%C3%A9?q=na%C3%AFve&x=1"), self.site.seen[0][0])

    def test_an_international_host_name_is_sent_as_its_ascii_form(self):
        self.serve({"/": (200, {"Content-Type": "text/plain"}, b"ok")})
        self.get("http://bücher.example/")
        self.assertEqual(self.site.seen[0][1]["Host"], "xn--bcher-kva.example")

    def test_a_server_that_hangs_up_is_an_error_not_a_crash(self):
        self.serve({})
        with patch.object(wf, "dial", lambda a, p, t: (_ for _ in ()).throw(ConnectionRefusedError("refused"))):
            with self.assertRaises(wf.FetchError) as caught:
                self.get()
        self.assertIn("did not answer", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
