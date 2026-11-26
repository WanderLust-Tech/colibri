"""Write c/tray.ico, the icon of colibri-tray.exe, from the very pixels the tray draws in the notification area.

    python tools/make_tray_ico.py            (from c/; writes tray.ico next to colibri_tray.rc)
    python tools/make_tray_ico.py --check    exit 1 when the file there is not what this would write

The icon is the green disc of an idle server at 16, 32, 48 and 64 pixels, as the BMP-style entries every Windows reads.
`tray.icon_pixels` is the one drawing; the C program has its own copy of it, which tests/test_tray_exe.py holds to it.
"""
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import tray  # noqa: E402

SIZES = (16, 32, 48, 64)


def straight(premultiplied):
    """Un-premultiply BGRA: an .ico holds straight alpha."""
    out = bytearray(premultiplied)
    for i in range(0, len(out), 4):
        alpha = out[i + 3]
        if 0 < alpha < 255:
            for c in range(3):
                out[i + c] = min(255, (out[i + c] * 255 + alpha // 2) // alpha)
    return bytes(out)


def entry_image(size):
    pixels = straight(tray.icon_pixels("idle", size))
    rows = [pixels[y * size * 4:(y + 1) * size * 4] for y in range(size)]
    xor = b"".join(reversed(rows))                                   # a DIB is stored bottom row first
    mask_row = ((size + 31) // 32) * 4
    mask = bytes(mask_row * size)                                    # all zero: the alpha channel does the shaping
    header = struct.pack("<IiiHHIIiiII", 40, size, size * 2, 1, 32, 0, len(xor) + len(mask), 0, 0, 0, 0)
    return header + xor + mask


def build():
    images = [entry_image(s) for s in SIZES]
    out = bytearray(struct.pack("<HHH", 0, 1, len(images)))
    offset = 6 + 16 * len(images)
    for size, image in zip(SIZES, images):
        out += struct.pack("<BBBBHHII", size, size, 0, 0, 1, 32, len(image), offset)
        offset += len(image)
    for image in images:
        out += image
    return bytes(out)


def main(argv):
    path = os.path.join(os.path.dirname(HERE), "tray.ico")
    data = build()
    if "--check" in argv:
        same = os.path.exists(path) and open(path, "rb").read() == data
        print("tray.ico is up to date" if same else "tray.ico is not what tools/make_tray_ico.py writes")
        return 0 if same else 1
    with open(path, "wb") as handle:
        handle.write(data)
    print(f"wrote {path} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
