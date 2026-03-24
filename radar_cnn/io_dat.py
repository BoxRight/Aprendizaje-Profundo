"""Load Glasgow INSHEP .dat: 4-line header + IQ beat samples."""

from __future__ import annotations

import numpy as np


def _parse_complex_line(s: str) -> complex:
    s = s.strip().replace("i", "j").replace("I", "j")
    return complex(s)


def read_glasgow_dat(path: str) -> tuple[np.ndarray, dict[str, float]]:
    """
    Glasgow `.dat` files use a **4-line header**, then either:

    1. **One complex per line** (MATLAB-style `a+bi`), as in the official
       Dataset_848 release — this is what `np.loadtxt` alone cannot read.
    2. **Flat interleaved Re, Im, Re, Im, …** floats (alternative export).

    Header lines: ``fc`` (Hz), ``Tsweep`` (ms), ``NTS`` (samples per chirp), ``Bw`` (Hz).

    Returns
    -------
    data_chirps : ndarray, shape (num_chirps, NTS), dtype complex64
    header_info : dict with fc_hz, tsweep_ms, tsweep_s, nts, bw_hz
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        header_lines: list[str] = []
        for _ in range(4):
            line = f.readline()
            if not line:
                raise ValueError(f"File too short (need 4 header lines + data): {path}")
            s = line.strip()
            if not s:
                raise ValueError(f"Empty header line in {path}")
            header_lines.append(s)

        data_lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if not data_lines:
        raise ValueError(f"No IQ samples after header: {path}")

    fc = float(header_lines[0])
    tsweep_ms = float(header_lines[1])
    nts = int(round(float(header_lines[2])))
    bw = float(header_lines[3])
    if nts <= 0:
        raise ValueError(f"Invalid NTS in header: {nts}")

    try:
        _parse_complex_line(data_lines[0])
        use_complex_lines = True
    except ValueError:
        use_complex_lines = False

    if use_complex_lines:
        iq_list = [_parse_complex_line(s) for s in data_lines]
        iq = np.array(iq_list, dtype=np.complex64)
    else:
        raw = np.array([float(x) for x in data_lines], dtype=np.float64)
        if len(raw) % 2 != 0:
            raise ValueError(
                f"Expected even number of IQ floats after header, got {len(raw)} in {path}"
            )
        iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)

    n_complex = iq.size
    if n_complex % nts != 0:
        raise ValueError(
            f"IQ length {n_complex} not divisible by NTS={nts} in {path}"
        )
    num_chirps = n_complex // nts
    data = iq.reshape(num_chirps, nts)
    header_info = {
        "fc_hz": fc,
        "tsweep_ms": tsweep_ms,
        "tsweep_s": tsweep_ms / 1000.0,
        "nts": float(nts),
        "bw_hz": bw,
    }
    return data, header_info
