# Neural network inputs: what to exclude and why (Pearl-aligned)

This list applies to **supervised** classifiers trained on the **Glasgow / INSHEP** style corpus (see [`01_dataset_inventory.md`](01_dataset_inventory.md)). **Targets** and **inputs** must not be confused.

Reasoning here is anchored in the **experimental** data-generating process (assignment, selection, processing)—what actually induces **\(P_{\text{exp}}(Y \mid X)\)**. That is **not** the same as the **phenomenon** DAG for real-world falls; see [**Phenomenon DAG vs experimental DAG**](03_causal_graph_falls.md#phenomenon-dag-vs-experimental-dag) in [`03_causal_graph_falls.md`](03_causal_graph_falls.md).

## 1. Intervened or protocol variables with **no or negligible variance**

If a variable is **fixed by the study** across all recordings you use, it is **constant** on your training support — **not identifiable** as a predictor (it does not reduce empirical risk) and can be **dropped**.

| Variable | Why useless in-sample |
|----------|------------------------|
| **Carrier frequency, chirp duration, bandwidth, samples/chirp** | Documented as fixed **5.8 GHz, 1 ms, 400 MHz, 128** ([`Readme.txt`](Readme.txt)). |
| **Same radar device and antenna model** | Single hardware regime. |
| **Same fixed `V` for ROI threshold** | If you always use Li et al.’s **V = 0.1**, that scalar is **constant** per pipeline. |

**Pearl note:** These are **not** “random” causes of \(Y\) in the dataset; they behave like **constants** in the structural equations. A neural net **cannot learn** a non-trivial dependence on a constant input.

## 2. **Intervention / assignment variables** that **duplicate the label** (leakage)

| Variable | Issue |
|----------|--------|
| **First digit `K` or `A01`–`A06` in filename** | **Encodes assigned activity** = **label** for multiclass ([`Readme.txt`](Readme.txt)), or trivially maps to fall vs non-fall. |
| **One-hot of session folder** if folders are **activity-pure** | Can **proxy** for label. |
| **Train table stratification** (e.g. “file ID in paper’s Table II”) | If used as a feature, **memorizes** split. |

**Pearl note:** The experiment **sets** \(A\) via instructions (`do(A)\)). The filename is **metadata about the intervention**, not a **sensor measurement** of an unobserved fall. Using it as input is **label leakage**, not causal signal from \(K \rightarrow M \rightarrow X\).

## 3. **Intervened kinematics** that are **not** radar-measured (collateral modalities)

| Variable | Issue |
|----------|--------|
| **X-IMU / wearable** channels from volunteer sheet | If present in **some** sessions only — **missingness** and **selection**; not pure radar. |
| **“Simulated fall style”** (trip vs slip) **if encoded in metadata** | Part of **protocol**, not echo physics. |

## 4. **Age and other demographics** — **exogenous but entangled with selection** (not “intervened” in the Pearl sense)

Age is **not** set by `do(·)` like carrier frequency or filename codes. It is an **exogenous** subject attribute. In this corpus it must still be treated as **disallowed as a model input**, for a reason stronger than “fairness” or “invariance.”

### Causal mechanism (aligned with [`03_causal_graph_falls.md`](03_causal_graph_falls.md))

- \(E_{\text{age}} \rightarrow A\): **Eligibility / safety policy** determines **who is assigned** the scripted-fall activity.
- \(A \rightarrow Y\): The **label** reflects **which script was performed**, not an independent clinical adjudication of a fall in the wild.
- Therefore **age indirectly determines whether a fall label exists** in the data: it is a **proxy for the selection mechanism** that defines \(Y\), not only for physiology.

In the dataset, **approximately**

\[
P(Y=\text{fall} \mid \text{older age}) \approx 0
\]

**not** because older adults do not fall in real life, but because **they were often excluded from fall recordings** for safety ([`Radar-based human activity recognition.txt`](Radar-based%20human%20activity%20recognition.txt), [`Readme.txt`](Readme.txt)). That is **censoring / selection induced by protocol**, not a property of radar physics.

If **age** (or a correlate available only from metadata) is fed as an input, the model can learn **“older → no fall”** and achieve **artificially high accuracy** on this dataset while encoding **dataset bias**, not motion or echo physics.

Formally, the exploitable path is:

\[
\text{Age} \rightarrow E_{\text{age}} \rightarrow A \rightarrow Y
\]

That is a path from **protocol construction to the label**, **not** from **body motion through the radar channel** to a fall. Using age as a feature is therefore **selection-induced leakage** relative to the causal question “does this **radar evidence** indicate a fall?”

| Variable | NN input? | Legitimate uses |
|----------|-------------|-----------------|
| **Age** | **No** (strong requirement: not optional “for invariance” only) | **Stratified evaluation**, **diagnosing missingness / selection bias**, reporting cohort statistics |
| **Height, gender, dominant hand** ([`Readme.txt`](Readme.txt)) | **No** as default for the same reason if they **track site, cohort, or eligibility** in this corpus; at minimum treat them like **non-radar** covariates and **ablate** to check for spurious shortcuts | Same as age where relevant |

**Pearl note:** Keep **intervened constants** (Section 1) and **assignment leakage** (Section 2) conceptually separate from this category: here demographics are **observable proxies for \(E_{\text{age}}\)** and **selection into \(Y\)**, not duplicate encodings of \(Y\) in the filename.

## 5. **Selection-induced leakage channels** (beyond age and filename)

These follow one pattern: a variable influences **which samples exist** or **how labels are assigned**, then becomes **spuriously predictive** of \(Y\) even though it is not on the causal path **motion → radar → features** (see [`03_causal_graph_falls.md`](03_causal_graph_falls.md)). They are **structurally implied** by site \(S\), assignment \(A\), eligibility, and protocol.

### 5.1 Site / regime (\(S\))

**Mechanism:** Different sites (lab vs NG Homes vs West Cumbria) differ in **which activities** were recorded and **whether falls** were collected ([`01_dataset_inventory.md`](01_dataset_inventory.md)). So \(S \rightarrow A \rightarrow Y\).

**Leakage channel:** Features that encode **environment**—static clutter, multipath, wall distance, noise floor, calibration offsets—can predict \(Y\) because **label mix co-varies with site**, not because they measure fall kinematics.

**Implication:** The model may learn **“this room / regime → fall trials exist here”** instead of **“this motion pattern → fall.”**

**Mitigation:** **Site-wise holdout**; background subtraction and robustness checks across sites.

### 5.2 Session / recording batch effects

**Mechanism:** Data are grouped by **session** (same day, setup, cohort). Often \(\text{Session} \rightarrow A \rightarrow Y\) via scheduling (which activities run when).

**Leakage channel:** Gain drift, amplitude scaling, ordering effects, **file order** (e.g. falls always last in a folder).

**Implication:** The model may **cluster sessions** instead of learning motion.

**Mitigation:** Split by **session** (or batch), not by isolated files; remove **temporal ordering** artifacts from features; shuffle at the protocol design level when collecting new data.

### 5.3 Subject identity (person ID / morphology)

**Mechanism:** Eligibility rules imply **some subjects never have fall labels**. So \(\text{Person} \rightarrow A \rightarrow Y\) (who gets which scripts).

**Leakage channel:** Radar also encodes **body morphology and motion style** (gait, limb length → Doppler structure). Identity is thus **entangled** with selection into the fall class.

**Implication:** The model can learn **“this person → no fall”** rather than **“this motion → fall.”** This is **selection bias**, not only overfitting.

**Mitigation:** **Subject-wise split** is non-negotiable for generalization claims; optional adversarial identity removal.

### 5.4 Repetition index (\(R\)) / trial ordering

**Mechanism:** Protocols often **order** repetitions (e.g. warm-up → daily activities → fall). Then \(R \rightarrow A \rightarrow Y\) in distribution.

**Leakage channel:** Fatigue, speed, or **implicit cues** from filename (`R1`–`R3`) or collection order.

**Implication:** **“Late repetition → fall likely”** without real physics.

**Mitigation:** Do not use **\(R\)** or order as inputs; check for **systematic ordering** in the corpus and report.

### 5.5 Scripted fall type / instruction variants

**Mechanism:** Only **certain fall types** are recorded (e.g. forward, controlled, cushioned on a mat); others are **absent** from support. \(\text{Fall type (script)} \rightarrow Y\) **by construction** for the fall class.

**Leakage channel:** Kinematic signatures specific to the **lab script** dominate the fall class.

**Implication:** The model may learn **“lab-style fall”** with **transportability failure** to spontaneous or backward falls—both **selection over kinematics** \(K\) and **external validity**.

**Mitigation:** Acknowledge **support** limits; test on held-out protocols if possible.

### 5.6 Preprocessing-dependent selection (ROI / thresholding)

**Mechanism:** \(P\) **selects** subsets of \(R\) (e.g. ROI keeps high-energy pixels). Falls often produce **different energy** than other activities.

**Leakage channel:** The model may latch onto **“amount of retained ROI** or **mask area**”—partly **pipeline-induced**, not purely physical.

**Implication:** **Fix \(P\) globally** across train/deploy and **sweep** \(P\) in ablations (see Section 6).

### 5.7 Missingness patterns (implicit labels)

**Mechanism:** Some **(site, age, activity)** combinations **do not exist** (e.g. older + fall; NG Homes subset **without** fall in [`Readme.txt`](Readme.txt)).

**Leakage channel:** **Absence** of a feature pattern can be **informative** of \(Y\) because **missingness is structured** by protocol.

**Implication:** **Dataset support** \(\neq\) real-world support; classifiers can exploit **holes** in the support.

**Mitigation:** Stratify reports by **which cells exist**; avoid features that are **defined only** when a class is present.

## 6. **Preprocessing artifacts** tied to \(do(P)\)

| Practice | Risk |
|----------|------|
| **ROI mask** from adaptive thresholding | Encodes **energy scaling** and **segmentation** — **portable** only if the same \(P\) is applied at deployment. Overlaps with **selection via \(P\)** (Section 5.6). |
| **Global normalization** using **statistics per file** that include **future** information | **Leakage** across time within a clip. |

## Recommended **inputs** vs **targets**

| Role | Suggested content |
|------|-------------------|
| **Input** | **Raw IQ** (optional) or **physics-based tensors** (range–time, Doppler–time spectrogram) **without** filename-derived codes; **pipeline** \(P\) fixed or ablated systematically. |
| **Target** | **\(Y\)** from **verified** label table (not from filename parsed into features). |
| **Evaluation groups** | **Subject-wise** holdout; **site-wise** and **session-aware** splits where possible (Sections 5.1–5.2). |

## Summary tables

### Selection-induced leakage (mechanism → action)

| Category | Example | Mechanism (path) | NN input? |
|----------|---------|------------------|-----------|
| Selection (site) | Room, clutter, multipath | \(S \rightarrow A \rightarrow Y\) | **No** (or control via **site-wise** split / held-out site) |
| Selection (subject) | Person ID, morphology entangled with gait | eligibility → who has fall labels | **No** |
| Selection (session) | Batch gain, ordering | Session \(\rightarrow A \rightarrow Y\) | **No**; split by session |
| Selection (repetition) | Trial index `R`, order | \(R \rightarrow A \rightarrow Y\) in distribution | **No** |
| Selection (demographics) | Age (and cohort-correlated fields) | Age \(\rightarrow E_{\text{age}} \rightarrow A \rightarrow Y\) | **No** |
| Selection (missingness) | Absent cells in support | structured \(\text{Missing} \leftrightarrow Y\) | Do not exploit **absence-as-signal**; report support |

### Quick categories (all categories)

| Category | Example | NN input? |
|----------|---------|-----------|
| Intervened constant | 5.8 GHz, 400 MHz | No (degenerate) |
| Assignment / label | `K`, `Axx` in filename | **No** (direct leakage) |
| Selection proxy | Age, site, person, session, \(R\) | **No** (Section 5) |
| Pipeline | STFT length, ROI mask params | Only if **fixed** or **shared** train/deploy; ablate \(P\) |
| Causal (motion → echo) | Spectrogram pixels, Doppler trajectories | **Yes** (primary) |
| Scripted-fall regime | Lab forward fall only | Not an input—**limitation** of \(Y\) and transportability |

**Boundary:** Variables that influence **which labels exist** or **how they are assigned** must not be treated as if they were **physics of motion → radar → features**. That is the line a **radar-only** fall model should respect.

This aligns with [`calcularCaracteristicasDoppler.py`](../calcularCaracteristicasDoppler.py): **features** derived from **micro-Doppler power** are **candidates** for inputs; **filename** and **fixed radar header** are **not**.
