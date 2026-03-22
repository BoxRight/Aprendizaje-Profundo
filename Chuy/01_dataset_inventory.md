# Dataset inventory — Glasgow Radar Signature (INSHEP)

This note merges the **INSHEP data sheet** ([`Readme.txt`](Readme.txt)), the **low-latency fall-detection paper** text, the **Scientific Reports HAR** paper, and **repository code**.

## Project and hardware

- **Project:** Intelligent RF Sensing for Falls and Health Prediction — **INSHEP** (EP/R041679/1), James Watt School of Engineering, University of Glasgow.
- **Radar:** Ancortek FMCW, **C-band ~5.8 GHz**, **400 MHz** chirp bandwidth, **1 ms** chirp / pulse repetition period, **~+18 dBm** output; **Yagi** TX/RX antennas **~+17 dBi**.
- **Digitization:** **128** complex samples per chirp (beat note), as stated in [`Readme.txt`](Readme.txt) and the papers.

## Collections over time (INSHEP)

Data were gathered across multiple sessions and venues ([`Readme.txt`](Readme.txt)):

| Block | When / where (summary) | DAT files | Subjects | Reps | Activities |
|------|-------------------------|-----------|----------|------|------------|
| 1 | Dec 2017, Glasgow lab | 360 | 20 | 3 | 6 (incl. fall) |
| 2 | Mar 2017 | 48 | 4 | 2 | 6 |
| 3 | Jun 2017 | 162 | 9 | 3 | 6 |
| 4 | Jul 2018, Glasgow common room | 288 | 16 | 3 | 6 |
| 5 | Feb 2019, Glasgow lab | 306 | 17 | 3 | 6 |
| 6a | Feb 2019, NG Homes Room 1 | 301 | (split tables) | 3 | **5** (no fall in this block) |
| 6b–c | NG Homes Rooms 2–3 | (same block 6) | older adults | 3 | 5 |
| 7 | Mar 2019, Age UK West Cumbria | 289 | 20 | 3 (exceptions noted) | 5 or 6 (see note) |

**Note:** Block 6 explicitly lists **five** activities (walking, sitting, standing, pick up, drink) — **no simulated fall** in that NG Homes subset. Block 7 text in [`Readme.txt`](Readme.txt) is partly duplicated across pages; treat activity list as **documented per session** when building splits.

**Global totals (papers):** **1754** motion captures, **72** participants, ages **21–98**, **~15.9 GB** ([`A Human Fall Detection low latency model for Doppler Radar sensors.v00_04_02.txt`](A%20Human%20Fall%20Detection%20low%20latency%20model%20for%20Doppler%20Radar%20sensors.v00_04_02.txt)). Summing INSHEP session file counts in [`Readme.txt`](Readme.txt) yields **1754**, consistent with that corpus.

## Activity semantics (six-class scheme)

Canonical **activity class index** (matches leading digit **K** and **A01–A06** in filenames — see below):

| Index | Activity (INSHEP / papers) |
|-------|----------------------------|
| 1 | Walking back and forth |
| 2 | Sitting down on a chair |
| 3 | Standing up from a chair |
| 4 | Picking up an object |
| 5 | Drinking water |
| 6 | Falling (simulated; **not all subjects**) |

**Older participants often did not perform falls** for safety ([`Readme.txt`](Readme.txt), papers). Fall class is **under-represented** (e.g. **198** fall files in the global 1754-file tally in the fall-detection paper text).

### Alignment with `Label_extract4.m`

[`../Sample_data_preprocessing_label_extraction/Label_extract4.m`](../Sample_data_preprocessing_label_extraction/Label_extract4.m) comments map **number2** to: 1 walk, 2 sit, 3 stand, **4 drink water, 5 pick**, 6 fall.

**INSHEP [`Readme.txt`](Readme.txt)** lists **K** as: 1 walk, 2 sit, 3 stand, **4 pick, 5 drink**, 6 fall.

So **activities 4 and 5 (drink vs pick) are swapped between the MATLAB comment block and the INSHEP data sheet.** Extraction code reads **`Axx`** from the filename; **the numeric code in `A01`–`A06` should be treated as ground truth** and reconciled with INSHEP’s table, not the comment order alone.

## Filename conventions

### INSHEP pattern: `KPXXAYYRZ.dat`

From [`Readme.txt`](Readme.txt):

- **K** — activity class **1–6** (first digit).
- **PXX** — subject ID **01, 02, …**
- **A01–A06** — activity label (redundant with K when files are consistent).
- **R1–R3** — repetition index.

Example from paper text: `1P01A01R2.dat` — person 1, walking, repetition 2.

### Variants in repo MATLAB

[`../PlotSpectrogram.m`](../PlotSpectrogram.m) uses `2P38A02R01.dat`: a **leading digit** (`2`) before **`P`** may indicate **session / subset / encoding** (not only K). Inventory should treat this as **another naming variant** to normalize when parsing.

### `Dataset_848` archive

[`../rejoin_dataset.sh`](../rejoin_dataset.sh) rebuilds **`Dataset_848.7z`**. The Glasgow record **848** matches [https://researchdata.gla.ac.uk/848/](https://researchdata.gla.ac.uk/848/) ([`Filiberto Lopez.Avance 3.txt`](Filiberto%20Lopez.Avance%203.txt)). Exact contents of the 7z (which subset of the 1754 files) should be confirmed by listing the archive after extraction.

## `.dat` file layout

### INSHEP / MATLAB format (dataset)

Per [`Readme.txt`](Readme.txt) and [`../PlotSpectrogram.m`](../PlotSpectrogram.m):

1. First **four** float values: **`fc` (Hz)**, **`Tsweep` in ms**, **`NTS`** (samples per chirp), **`Bw` (Hz)**.
2. Remaining samples: **complex beat-note sequence** (I/Q as real/imaginary parts in MATLAB `textscan` / `reshape`).

Papers add: **5 s** or **10 s** recordings → **640,000** or **1,280,000** complex samples after header, depending on activity (walking 10 s vs others 5 s) ([`A Human Fall Detection low latency model for Doppler Radar sensors.v00_04_02.txt`](A%20Human%20Fall%20Detection%20low%20latency%20model%20for%20Doppler%20Radar%20sensors.v00_04_02.txt)).

### Python script in repo

[`../graficarDatosRadar.py`](../graficarDatosRadar.py) expects **only** ASCII complex lines (e.g. `2200+1869j`) **without** the four-float header. Use it only on files in that format; **INSHEP `.dat` files need header stripping** to match that pipeline.

## Subject metadata

[`Readme.txt`](Readme.txt) includes **age, height, gender, dominant hand** per subject ID with **`n/a`** where missing. Use for **stratified analysis**, not as causal radar inputs (see [`04_nn_inputs_exclusions.md`](04_nn_inputs_exclusions.md)).

## Open questions

1. **Exact mapping** of `2P38…` leading digit vs. INSHEP **K** across all releases.
2. **Contents of `Dataset_848.7z`** vs. full 1754-file corpus.
3. **West Cumbria / NG Homes** activity counts: confirm fall presence per session from filenames or README on the archive.
