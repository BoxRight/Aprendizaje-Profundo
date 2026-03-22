# Literature synthesis — radar HAR/fall detection and Glasgow data

Cross-links the text sources in [`Chuy/`](.) to the **INSHEP / Glasgow Radar Signature** design. See [`01_dataset_inventory.md`](01_dataset_inventory.md) for raw numbers and filename rules.

## INSHEP data sheet ([`Readme.txt`](Readme.txt))

- **Aim:** Contactless monitoring for **activity patterns** and **critical events (falls)**; privacy vs. cameras.
- **Protocol:** “Snapshot” activities performed **separately**; **simulated frontal fall** only where **lab control and safety** allowed — **subset of subjects**.
- **Hardware:** Fixed Ancortek FMCW parameters (5.8 GHz, 400 MHz, 1 ms, 128 samples/chirp, Yagi antennas).
- **Implication for modeling:** Falls are **scripted** and **not uniformly available** across age groups or venues; **session and site** differ (lab, common room, NG Homes, West Cumbria).

## Volunteer information sheet ([`Information Sheet for Volunteers.txt`](Information%20Sheet%20for%20Volunteers.txt))

- Broader **Feb 2019** protocol lists many daily tasks; **lab-only** extras include **simulated trip + fall on soft mat** (controlled movement, spotters).
- **Wearable X-IMU** may be used in some experiments — **another modality** not assumed in pure-radar `.dat` classifiers unless synchronized subsets are used.
- **Causal note:** The sheet describes **how participants were instructed** (intervention on **motion script**), which does not appear in radar pixels but shapes the **distribution of kinematics**.

## Li et al., *Scientific Reports* (2023) — [`Radar-based human activity recognition.txt`](Radar-based%20human%20activity%20recognition.txt)

- **Dataset:** Same **University of Glasgow Radar Signature** corpus: **1754** captures, **72** people, **six activities** including fall; **imbalanced** because older adults **omit fall** recordings.
- **Processing:** Range–time → **MTI-style clutter rejection** (4th-order Butterworth high-pass, cutoff **0.0075** normalized in slow-time), **STFT** micro-Doppler with **0.2 s** Hamming window, **95% overlap**; walking split into **two 5 s** segments to match duration.
- **Method focus:** **Adaptive thresholding** builds an **ROI mask** on spectrograms; features on **magnitude, phase, unwrapped phase**; **hierarchical** multi-stage classifier; compares **training time, inference time, memory** to deep models.
- **Tie-in:** Shows that **engineering choices** (threshold **V = 0.1**, mask features, SVM/NB fusion) are **strong interventions on the feature pipeline** — stable within benchmark but **not intrinsic** to physics of falling.

## “Maximum velocity” / SVM fall detection — [`A Human Fall Detection low latency model for Doppler Radar sensors.v00_04_02.txt`](A%20Human%20Fall%20Detection%20low%20latency%20model%20for%20Doppler%20Radar%20sensors.v00_04_02.txt)

- **Task:** Binary **fall vs non-fall** using **maximum radial speed** from Doppler spectrogram + **SVM** (low-resource motivation vs. heavy DL).
- **Dataset facts repeated:** FMCW parameters, file counts per activity (**198** fall files), **filename structure** `K P XX A YY R Z` (activity digit + `P` + subject + `A` + activity code + `R` + rep).
- **Feature pipeline:** Reshape to **chirps × fast-time samples**, Hamming FFT → velocity–time; second FFT → **spectrogram**; extract **maximum velocity** feature.
- **Train split:** Random **154** files (stratified table: **17** fall files among **154**) — sample-size formula using fall proportion **p ≈ 0.1129**.
- **Tie-in:** Uses **single scalar** derived from spectrogram — **compresses** micro-Doppler; good for latency but **collapses** information that may separate **confusable activities** (e.g. fast non-fall motions).

## Feature extraction notes ([`Feature Extraction.txt`](Feature%20Extraction.txt))

- Standard **FMCW** range and Doppler relations, **STFT/micro-Doppler** path for limb modulation.
- Frames of **N** chirps (e.g. 128, 256) align with choices in [`../graficarDatosRadar.py`](../graficarDatosRadar.py) / common practice — **hyperparameters** for NN or classical pipelines.

## Progress report — [`Filiberto Lopez.Avance 3.txt`](Filiberto%20Lopez.Avance%203.txt)

- Survey of **radar fall-detection** literature by domain (micro-Doppler, range–Doppler, etc.).
- **Dataset pointers:** mmFall, PeerJ multimodal, PARrad, and **Glasgow / Nature 2023** + record **[848](https://researchdata.gla.ac.uk/848/)**.
- **Research question:** **Low-latency** algorithms suitable for **home / edge** deployment.

## Consolidated implications for causal analysis

| Theme | Source | Implication |
|-------|--------|-------------|
| Scripted falls | INSHEP README, volunteer sheet | **Not** spontaneous falls; **kinematics distribution is lab-shaped**. |
| Age-based exclusion from fall class | Papers, README | **Selection mechanism**: fall label **missing** for older cohort — confounds naive “age → signature” if not modeled. |
| Fixed radar parameters | README | **Near-constant** across files — little **in-sample** variance (see [`04_nn_inputs_exclusions.md`](04_nn_inputs_exclusions.md)). |
| Preprocessing | Li et al. | **do(Algorithm)** on raw IQ — defines what the NN sees; must not confuse with **latent fall**. |

Next: [`03_causal_graph_falls.md`](03_causal_graph_falls.md) encodes these as a DAG; [`04_nn_inputs_exclusions.md`](04_nn_inputs_exclusions.md) lists non-causal or leaky inputs.
