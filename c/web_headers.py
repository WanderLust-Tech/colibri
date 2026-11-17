"""Headers the gateway and the router send with what a browser may show or run, and a second check on cross-site requests.

The dashboard is a static page served by the same process as the API. Nothing in it builds markup from model output (the
Markdown component does not use innerHTML), so these are defence in depth: if a script ever did get into the page, the policy
keeps it from loading more code or framing the page, and if a page on another site drives the API, the browser's own
`Sec-Fetch-Site` says so even where the origin allow-list would not have been looked at (a simple request is carried out before
the browser decides the answer may not be read). The key is still what protects the API; see docs/THREAT_MODEL.md.
"""

# Sent with every response.
BASELINE = (
    ("X-Content-Type-Options", "nosniff"),         # a JSON or text answer is never run as a script or a page
    ("Referrer-Policy", "no-referrer"),            # a link out of the dashboard does not carry the server's address
)
BASELINE_NAMES = frozenset(name.lower() for name, _ in BASELINE)

# What the dashboard loads: its own script, style, font and icon files from the same address; pictures and recordings it makes
# itself (data: and blob:); and the API, which may be at another address when the dashboard is pointed at one (the Connection
# page), hence http: and https: for connect-src. No plug-ins, no <base>, no framing by another page.
CSP = "; ".join((
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self' 'unsafe-inline'",           # React writes style attributes
    "img-src 'self' data: blob:",
    "media-src 'self' data: blob:",
    "font-src 'self' data:",
    "connect-src 'self' http: https:",
    "object-src 'none'",
    "base-uri 'none'",
    "form-action 'self'",
    "frame-ancestors 'none'",
))

# Added to the dashboard's own files.
DOCUMENT = (
    ("Content-Security-Policy", CSP),
    ("X-Frame-Options", "DENY"),                    # the same rule for a browser that does not read frame-ancestors
)


def cross_site_refused(path, site, origin, allowed_origins):
    """True for a browser request to the API that comes from another site and from an origin the operator did not allow.

    Only a browser sends `Sec-Fetch-Site`; curl, the SDKs and the editor extension send none, and nothing changes for them. The
    dashboard talking to its own server is `same-origin`, and a page on an allowed origin (`--cors-origin`) is let through.
    """
    if not path.startswith("/v1/") or (site or "").lower() != "cross-site":
        return False
    if "*" in allowed_origins or (origin and origin in allowed_origins):
        return False
    return True
