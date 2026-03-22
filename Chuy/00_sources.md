# Sources index (`Chuy/`)

This folder holds **plain-text exports** of PDFs and the INSHEP data sheet, plus the analysis markdown files produced from them. Paths below are relative to the project root [`Aprendizaje Profundo/`](../).

## Primary text sources (in `Chuy/`)

| File | Contents |
|------|----------|
| [`Readme.txt`](Readme.txt) | INSHEP (EP/R041679/1) data sheet: collections (2017–2019), radar hardware, **`KPXXAYYRZ`** naming, per-session file counts and subject tables, `.dat` header layout. |
| [`A Human Fall Detection low latency model for Doppler Radar sensors.v00_04_02.txt`](A%20Human%20Fall%20Detection%20low%20latency%20model%20for%20Doppler%20Radar%20sensors.v00_04_02.txt) | Low-latency fall detection using **maximum Doppler velocity** + SVM; Glasgow dataset statistics, filename grammar, train split table. |
| [`Radar-based human activity recognition.txt`](Radar-based%20human%20activity%20recognition.txt) | Li et al., *Scientific Reports* (2023): adaptive thresholding ROI, hierarchical HAR, same Glasgow dataset preprocessing notes. |
| [`Feature Extraction.txt`](Feature%20Extraction.txt) | FMCW formulas, range–Doppler–micro-Doppler processing outline. |
| [`Information Sheet for Volunteers.txt`](Information%20Sheet%20for%20Volunteers.txt) | Feb 2019 ethics sheet: activity list, **simulated trip + fall on mat** (lab only), optional wearable X-IMU. |
| [`Filiberto Lopez.Avance 3.txt`](Filiberto%20Lopez.Avance%203.txt) | Progress report: radar FDS survey, links to external datasets and **Glasgow record 848**. |

## Official dataset and publications (URLs)

- **University of Glasgow research data (record 848):** [https://researchdata.gla.ac.uk/848/](https://researchdata.gla.ac.uk/848/) — landing page for the Radar Signature corpus referenced across papers.
- **HAR / adaptive thresholding paper:** [https://doi.org/10.1038/s41598-023-30631-x](https://doi.org/10.1038/s41598-023-30631-x) (Li et al., *Scientific Reports* 2023).
- Optional Springer link cited in `Filiberto Lopez.Avance 3.txt` for related reading (verify if still current).

## Repository code (processing and naming)

| Path | Role |
|------|------|
| [`../rejoin_dataset.sh`](../rejoin_dataset.sh) | Rebuilds `Dataset_848.7z` from `Dataset_848.7z.part*`. |
| [`../Sample_data_preprocessing_label_extraction/Label_extract4.m`](../Sample_data_preprocessing_label_extraction/Label_extract4.m) | Parses **`Pxx` / `Axx` / `Rxx`** from filenames; maps activity codes to labels (see [`01_dataset_inventory.md`](01_dataset_inventory.md) for INSHEP alignment). |
| [`../PlotSpectrogram.m`](../PlotSpectrogram.m), [`../DataProcessingExample.m`](../DataProcessingExample.m) | Read `.dat` with **4-float header** then IQ samples. |
| [`../graficarDatosRadar.py`](../graficarDatosRadar.py) | Standalone plot path assuming **no header** (different from INSHEP `.dat` layout). |
| [`../calcularCaracteristicasDoppler.py`](../calcularCaracteristicasDoppler.py) | Hand-crafted micro-Doppler fall-oriented features. |

## Optional duplicate PDFs (outside `Chuy/`)

Original PDFs may exist under project folders such as `Documentos 2025 - FLopez/` or `Metadata/`. The **`Chuy/*.txt`** copies are the canonical working inputs for this documentation set.

## Derived deliverables (this folder)

| File | Description |
|------|-------------|
| [`01_dataset_inventory.md`](01_dataset_inventory.md) | Consolidated dataset facts and file format notes. |
| [`02_literature_synthesis.md`](02_literature_synthesis.md) | Paper summaries tied to protocol and interventions. |
| [`03_causal_graph_falls.md`](03_causal_graph_falls.md) | Phenomenon vs experimental DAGs, refined experimental DAG, transportability, selection. |
| [`04_nn_inputs_exclusions.md`](04_nn_inputs_exclusions.md) | What not to feed a neural network and why. |
