"""Map codec: the serialized form that goes over the link.

The dashboard renders what it receives from this codec — not a separate, roomier JSON path.
So the bandwidth figure on screen is the bandwidth actually being used, and a decode bug
shows up as a broken map rather than as a silently optimistic number.

Layout, little-endian:

    magic    4s   b"PRH1"
    version  u8
    flags    u8    bit0 = zlib-compressed body
    frame    u32
    n_cells  u32
    quant    f32   metres per x/y unit (half a minimum cell, so every level's
                   centre lands exactly on a step)
    root     f32   metres, root cell size (level 0)
    json_len u32
    ---- struct of arrays, n_cells each ----
    x        i16   units of quant from the sensor
    y        i16
    z        i16   centimetres
    level    u8    cell size = root / 2**level
    class    u8
    conf     u8    0..255
    driver   u8    which rule set this cell's resolution

    The three int16 arrays lead so that every typed-array view in the browser lands on a
    two-byte boundary whatever n is. Interleaving them with the u8 arrays would make the
    decoder copy.
    ---- json tail ----
    stats + detections, json_len bytes of UTF-8
"""
from __future__ import annotations

import json
import struct
import zlib

import numpy as np

from . import config as C
from .grid_engine import Grid

MAGIC = b"PRH1"
VERSION = 1
HEADER = struct.Struct("<4sBBIIffI")
FLAG_COMPRESSED = 0x01


def encode(grid: Grid, frame_id: int, tail: dict, compress: bool = True,
           level: int = 6) -> tuple[bytes, dict]:
    n = len(grid)
    tail_bytes = json.dumps(tail, separators=(",", ":")).encode("utf-8")

    xq = np.rint(grid.x / C.QUANT_M).astype("<i2")
    yq = np.rint(grid.y / C.QUANT_M).astype("<i2")
    zq = np.clip(np.rint(grid.z * 100.0), -32768, 32767).astype("<i2")
    cq = np.clip(np.rint(grid.conf * 255.0), 0, 255).astype("<u1")

    body = b"".join([
        xq.tobytes(), yq.tobytes(), zq.tobytes(),
        grid.level.astype("<u1").tobytes(),
        grid.cls.astype("<u1").tobytes(),
        cq.tobytes(),
        grid.driver.astype("<u1").tobytes(),
        tail_bytes,
    ])
    head = HEADER.pack(MAGIC, VERSION, FLAG_COMPRESSED if compress else 0,
                       frame_id, n, C.QUANT_M, C.ROOT_SIZE_M, len(tail_bytes))
    raw = head + body
    wire = head + zlib.compress(body, level) if compress else raw

    info = {
        "raw_bytes": len(raw),
        "wire_bytes": len(wire),
        "compression_ratio": round(len(raw) / len(wire), 2) if wire else None,
        "bytes_per_cell_wire": round(len(wire) / n, 2) if n else None,
        "tail_bytes": len(tail_bytes),
    }
    return wire, info


def decode(buf: bytes) -> dict:
    """Reference decoder. The browser implements the same layout; this one keeps the
    format honest in the test suite."""
    magic, version, flags, frame, n, quant, root, json_len = HEADER.unpack_from(buf, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported version {version}")
    body = buf[HEADER.size:]
    if flags & FLAG_COMPRESSED:
        body = zlib.decompress(body)

    o = 0
    def take(dtype, count, itemsize):
        nonlocal o
        a = np.frombuffer(body, dtype=dtype, count=count, offset=o)
        o += count * itemsize
        return a

    x = take("<i2", n, 2).astype(np.float32) * quant
    y = take("<i2", n, 2).astype(np.float32) * quant
    z = take("<i2", n, 2).astype(np.float32) / 100.0
    lvl = take("<u1", n, 1)
    cls = take("<u1", n, 1)
    conf = take("<u1", n, 1).astype(np.float32) / 255.0
    drv = take("<u1", n, 1)
    tail = json.loads(body[o:o + json_len].decode("utf-8")) if json_len else {}

    return {"frame": frame, "n_cells": n, "quant": quant, "root": root,
            "x": x, "y": y, "level": lvl, "z": z, "cls": cls, "conf": conf,
            "driver": drv, "tail": tail}


def link_budget(wire_bytes: int, hz: float) -> dict:
    """What this map costs on a constrained tactical link."""
    bps = wire_bytes * 8 * hz
    return {
        "kb_per_frame": round(wire_bytes / 1024.0, 1),
        "kbps": round(bps / 1000.0, 1),
        "mbps": round(bps / 1e6, 3),
        # Reference points for a jam-constrained link, so the number means something.
        "fits_256kbps_link": bps <= 256_000,
        "fits_1mbps_link": bps <= 1_000_000,
    }
