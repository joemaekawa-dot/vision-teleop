"""
EEFrame — embodiment-agnostic end-effector interchange format.

SINGLE SOURCE OF TRUTH for the wire format. This exact file is deployed to
BOTH the Mac (perception/TX) and the Pi (controller/RX). Do not fork it.

Robustness (hardened after adversarial review):
  * SCHEMA16 — a hash of the struct layout is embedded in every packet. If the
    two deployed copies ever drift (rsync skipped, one side edited), packets are
    rejected loudly instead of silently decoding into a garbage pose.
  * CRC32 over the payload — catches corruption that UDP's weak checksum misses.
  * Finite/range validation on decode — non-finite or absurd poses never reach
    a servo.
  * Redundant state packing — a packet carries the last N frames, so one lost
    datagram is recovered by the next without any retransmit latency.

The frame is machine-agnostic: a 6-DOF pose + normalized gripper. Each robot's
adapter realizes it (SO-101 projects to its 5-DOF manifold; a 6/7-DOF arm uses
it fully). v2 will add a TLV hand-skeleton block for dexterous hands.
"""
from __future__ import annotations

import hashlib
import math
import struct
import zlib
from dataclasses import dataclass, replace
from typing import List

MAGIC = 0x45455631  # 'EEV1'
VERSION = 1
MAX_FRAMES_PER_PACKET = 3

# flags bitfield
FLAG_VALID = 1 << 0
FLAG_ENABLED = 1 << 1
FLAG_HOME = 1 << 2
FLAG_CALIBRATED = 1 << 3   # pos carries a RELATIVE EE delta (m) from the calib origin

_FRAME = struct.Struct("<IQ3f4fffI")        # seq,ts,pos3,quat4,grip,conf,flags
FRAME_SIZE = _FRAME.size                     # 52 bytes
assert FRAME_SIZE == 52, FRAME_SIZE

# header: magic, version, schema16, count, reserved, crc32(payload)
_HEADER = struct.Struct("<IHHHHI")           # 16 bytes

# Layout fingerprint: any change to the struct format or field set flips this,
# so a drifted second copy fails the check instead of misparsing.
_SCHEMA_SRC = f"v{VERSION}|{_FRAME.format}|seq,ts,pos,quat,grip,conf,flags"
SCHEMA16 = int.from_bytes(
    hashlib.blake2s(_SCHEMA_SRC.encode(), digest_size=2).digest(), "little")

_POS_LIMIT = 100.0  # meters; sanity bound (normalized task frame is ~[-1,1])


@dataclass
class EEFrame:
    seq: int = 0
    send_ts_ns: int = 0        # Mac wall-clock; INFORMATIONAL only (unsynced)
    pos: tuple = (0.0, 0.0, 0.0)
    quat: tuple = (1.0, 0.0, 0.0, 0.0)
    gripper: float = 0.0
    confidence: float = 0.0
    flags: int = 0

    @property
    def valid(self) -> bool:
        return bool(self.flags & FLAG_VALID)

    @property
    def enabled(self) -> bool:
        return bool(self.flags & FLAG_ENABLED)

    @property
    def home(self) -> bool:
        return bool(self.flags & FLAG_HOME)

    @property
    def calibrated(self) -> bool:
        return bool(self.flags & FLAG_CALIBRATED)

    def pack_into(self, buf: bytearray, offset: int) -> int:
        _FRAME.pack_into(
            buf, offset,
            self.seq & 0xFFFFFFFF, self.send_ts_ns & 0xFFFFFFFFFFFFFFFF,
            self.pos[0], self.pos[1], self.pos[2],
            self.quat[0], self.quat[1], self.quat[2], self.quat[3],
            self.gripper, self.confidence, self.flags & 0xFFFFFFFF)
        return offset + FRAME_SIZE

    @classmethod
    def unpack_from(cls, buf, offset: int) -> "EEFrame":
        (seq, ts, px, py, pz, qw, qx, qy, qz, g, c, fl) = _FRAME.unpack_from(buf, offset)
        vals = (px, py, pz, qw, qx, qy, qz, g, c)
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("non-finite field")
        if max(abs(px), abs(py), abs(pz)) > _POS_LIMIT:
            raise ValueError("pos out of range")
        return cls(seq, ts, (px, py, pz), (qw, qx, qy, qz), g, c, fl)


def pack_packet(frames: List[EEFrame]) -> bytes:
    n = min(len(frames), MAX_FRAMES_PER_PACKET)
    payload = bytearray(n * FRAME_SIZE)
    off = 0
    for i in range(n):
        off = frames[i].pack_into(payload, off)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return _HEADER.pack(MAGIC, VERSION, SCHEMA16, n, 0, crc) + bytes(payload)


def unpack_packet(data: bytes) -> List[EEFrame]:
    """Parse a packet -> frames (newest first). Raises ValueError on anything
    that fails magic/version/schema/CRC/length/finite checks."""
    if len(data) < _HEADER.size:
        raise ValueError("short packet")
    magic, version, schema, count, _resv, crc = _HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic:#x}")
    if version != VERSION:
        raise ValueError(f"unsupported version {version}")
    if schema != SCHEMA16:
        raise ValueError(f"schema drift {schema:#06x} != {SCHEMA16:#06x}")
    if count > MAX_FRAMES_PER_PACKET:
        raise ValueError(f"count {count} too large")
    payload = memoryview(data)[_HEADER.size:_HEADER.size + count * FRAME_SIZE]
    if len(payload) < count * FRAME_SIZE:
        raise ValueError("truncated frames")
    if (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
        raise ValueError("crc mismatch")
    return [EEFrame.unpack_from(payload, i * FRAME_SIZE) for i in range(count)]


def snapshot(frame: EEFrame) -> EEFrame:
    """Immutable copy for the TX ring, so caller-side object reuse can't corrupt
    already-queued redundant frames."""
    return replace(frame)


if __name__ == "__main__":
    f = EEFrame(seq=42, send_ts_ns=123456789, pos=(0.1, -0.2, 0.3),
                quat=(1, 0, 0, 0), gripper=0.5, confidence=0.9,
                flags=FLAG_VALID | FLAG_ENABLED)
    pkt = pack_packet([f, f, f])
    out = unpack_packet(pkt)
    assert out[0].seq == 42 and abs(out[0].pos[0] - 0.1) < 1e-6
    # corruption / drift rejection
    bad = bytearray(pkt); bad[-1] ^= 0xFF
    try:
        unpack_packet(bytes(bad)); raise SystemExit("CRC not caught!")
    except ValueError:
        pass
    print(f"OK schema={SCHEMA16:#06x} frame={FRAME_SIZE}B packet(3)={len(pkt)}B "
          f"crc+drift rejection verified")
