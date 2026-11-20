# Inventory Image Tagging via Handcrafted Features and Deep Learning

**Project:** Inventory Image Tagging via Handcrafted Features and Deep Learning (Phase 1 + Phase 2)  
**Course Project:** Project 11 | Team T112 | Section D (6 Members)  
**Dataset:** [Fashion-MNIST (Official Zalando Dataset)](https://github.com/zalandoresearch/fashion-mnist) — $70{,}000$ Grayscale Images ($28 \times 28$) across 10 Clothing Categories  
**Repository:** [`anantjain-afk/Image-Tagging`](https://github.com/anantjain-afk/Image-Tagging) (Owner & Sole PR Merger: **Anant Jain**)  
**Status:** **Completed (Phase 1 Applied Machine Learning + Phase 2 Deep Learning & Ablation)**  

---

## 1. Executive Summary & The Hybrid ML Logic

E-commerce catalogue tagging requires automated, high-throughput classification of garment photographs into inventory categories. This repository documents a structured, two-phase empirical investigation addressing the central research question:

> **Core Research Question:**  
> *Can fixed, handcrafted visual descriptors (Raw Pixels and HOG) adequately classify apparel imagery, or does fine-grained upper-body category confusion necessitate hierarchical visual representations learned end-to-end by Convolutional Neural Networks?*

To answer this, our project follows a rigorous, staged experimental roadmap:

1. **Phase 1: Applied Machine Learning (AML) Baseline:**
   - Evaluated 4 classical classifier families (Multinomial Logistic Regression, $k$-Nearest Neighbors, Random Forests, and Linear Support Vector Machines) across 2 feature representations ($784$-d normalized raw pixels vs.\ $324$-d multi-scale Histograms of Oriented Gradients), yielding **8 baseline configurations**.
   - **The Discovery of "The Gap":** While overall test accuracy reached $87.43\%$ (Random Forest), class-conditional error analysis uncovered a universal bottleneck: **Shirt (Class 6) recall collapsed to $50.7\%$--$62.6\%$**, with $35.8\%$ of Shirt test samples erroneously leaking into adjacent upper-body categories (T-shirt/top, Pullover, and Coat).

2. **Phase 2: Deep Learning Architectures & Transfer Learning:**
   - Addressed the representation deficit by engineering and evaluating 3 deep convolutional models:
     - **SimpleCNN:** 2-block convolutional baseline ($421{,}738$ parameters).
     - **DeepCNN:** 3-block resolution-tailored convolutional architecture ($390{,}634$ parameters).
     - **RegNet-Y-400MF:** Adapted ImageNet transfer learning backbone ($3{,}906{,}978$ parameters).
   - **The Resolution of The Gap:** DeepCNN achieved **$92.21\%$ test accuracy and $92.16\%$ Macro $F_1$**, elevating Shirt recall to **$73.2\%$** ($+13.9$ percentage points over Random Forest, $+10.6$~pp over the highest classical model) and reducing upper-body confusion from $35.8\%$ down to $23.6\%$.
   - **Controlled Ablation Study:** A 5-variable controlled ablation study isolated **Batch Normalization** ($-6.0$~pp recall when removed) and **Data Augmentation** ($-4.7$~pp recall when removed) as the vital regularizers enabling recovery of fine-grained collar and placket features.

---

## 2. Master Cross-Phase Benchmark (All 11 Configurations)

All models are evaluated on the exact same held-out $10{,}000$-image test set using a strictly reproducible stratified split ($48{,}000$ train, $12{,}000$ validation, $10{,}000$ test, `seed=42`).

| # | Model Architecture | Feature Representation | Trainable Parameters | Inference Latency | Test Accuracy | Test Macro $F_1$ | Shirt (Class 6) Recall | Phase / Status |
|---|---|---|---|---|---|---|---|---|
| 1 | **Linear SVM** | Raw Pixels ($784$-d) | 7,850 | 0.001 ms | 83.80% | 83.57% | 50.7% | Phase 1 (AML) |
| 2 | **Linear SVM** | HOG ($324$-d) | 3,250 | 0.001 ms | 84.23% | 84.06% | 54.6% | Phase 1 (AML) |
| 3 | **Multinomial LogReg** | Raw Pixels ($784$-d) | 7,850 | 0.001 ms | 84.32% | 84.24% | 57.1% | Phase 1 (AML) |
| 4 | **Multinomial LogReg** | HOG ($324$-d) | 3,250 | <0.001 ms | 84.55% | 84.49% | 57.8% | Phase 1 (AML) |
| 5 | **Random Forest** | Raw Pixels ($784$-d) | 1,720,000 | 0.026 ms | **87.43%** | **87.31%** | 59.3% | Phase 1 (AML Best) |
| 6 | **Random Forest** | HOG ($324$-d) | 850,000 | 0.018 ms | 84.35% | 84.30% | 59.4% | Phase 1 (AML) |
| 7 | **$k$-NN ($k=5$)** | Raw Pixels ($784$-d) | Non-parametric (179 MB) | 1.840 ms | 85.41% | 85.40% | 60.2% | Phase 1 (AML) |
| 8 | **$k$-NN ($k=5$)** | HOG ($324$-d) | Non-parametric (74 MB) | 0.920 ms | 83.84% | 83.85% | **62.6%** | Phase 1 (AML) |
| 9 | **SimpleCNN (2-Block)** | Learned 2D Tensors | 421,738 | 0.024 ms | 91.14% | 91.08% | 70.9% | Phase 2 (DL) |
| 10 | **DeepCNN (3-Block)** | **Learned 2D Tensors** | **390,634** | **0.029 ms** | **92.21%** | **92.16%** | **73.2%** | **Phase 2 (Pareto-Optimal)** |
| 11 | **RegNet-Y-400MF** | ImageNet Transfer + Fine | 3,906,978 | 0.121 ms | 91.46% | 91.42% | **73.4%** | Phase 2 (Transfer) |

---

## 3. Key Findings & Scientific Insights

### 3.1. Phase 1 AML Findings: The Universal Shirt Deficit
- **The Classical Ceiling:** Depth-regularized Random Forest (200 trees, $\text{max\_depth}=20$) achieved the highest classical performance ($87.43\%$ accuracy, $87.31\%$ Macro $F_1$).
- **The Representation Deficit:** Across all 8 classical models, Shirt recall was constrained between $50.7\%$ (SVM Pixels) and $62.6\%$ ($k$-NN HOG), while Trouser ($96.7\%$) and Bag ($97.3\%$) achieved near-perfect accuracy.
- **Physical Causes of Failure:**
  1. *Destruction of Spatial Topology:* Flattening 2D images to 1D vectors destroys local adjacency; button plackets and collar edges become isolated scalar features.
  2. *HOG Spatial Averaging:* At $28 \times 28$ resolution, a shirt collar spans roughly $3 \times 3$ pixels. HOG's $7 \times 7$ cell integration averages these localized gradient spikes, smoothing away discriminative cues.
  3. *Silhouette Equivalence:* T-shirts, Pullovers, Coats, and Shirts share identical torso bounding boxes; classical linear boundaries and axis-aligned tree splits cannot separate them without fine interior features.

### 3.2. Phase 2 Deep Learning Recovery: Closing the Representation Gap
- **Superiority of Learned Representations:** Convolutions preserve 2D topology and extract hierarchical features (edges $\to$ contours $\to$ garment structure), boosting catalogue Macro $F_1$ from $87.31\%$ to **$92.16\%$**.
- **Quantified Shirt Recovery:**
  - Random Forest misclassifies **$407 / 1{,}000$** Shirts ($40.7\%$ error rate).
  - DeepCNN misclassifies only **$268 / 1{,}000$** Shirts ($26.8\%$ error rate).
  - **Over 130 previously lost garment images recovered per 1,000 items.**
- **Upper-Body Misclassification Collapse:**
  - Shirt $\to$ Pullover leakage drops from $12.0\%$ to **$6.1\%$** (halved).
  - Shirt $\to$ Coat leakage drops from $9.0\%$ to **$7.0\%$**.
  - Shirt $\to$ T-shirt leakage drops from $14.8\%$ to **$10.5\%$**.
  - Total upper-body confusion contracts from **$35.8\%$ down to $23.6\%$**.

### 3.3. The Architectural Parameter Paradox
Despite having an additional convolutional block and doubling channel capacity ($128$ filters vs.\ $64$), **DeepCNN contains $31{,}104$ fewer parameters than SimpleCNN ($391\text{K}$ vs.\ $422\text{K}$)**.  
*Why?* The third max-pooling stage reduces the feature map from $7 \times 7$ to $3 \times 3$, shrinking the flattened dense projection input from $3{,}136$ to $1{,}152$. This shrinks the dense linear projection matrix by $26.3\%$ ($294{,}912$ vs.\ $401{,}408$ weights), delivering higher accuracy with smaller model weight footprint.

### 3.4. Controlled 5-Variable Ablation Study

| Ablation Condition | Key Modification | Test Accuracy | Macro $F_1$ | Shirt Recall | Key Takeaway |
|---|---|---|---|---|---|
| **Full DeepCNN (Base)** | 3 Blocks + BN + Drop + Aug | 90.26% | 90.27% | **72.8%** | Controlled baseline checkpoint |
| **No Dropout** | Conv & Dense Dropout removed | **91.93%** | **91.88%** | 72.0% | Macro F1 rises on easy classes, but Shirt recall drops ($-0.8$ pp) |
| **No BatchNorm** | All BatchNorm2d layers removed | 89.90% | 89.83% | **66.8%** | **Severe collapse ($-6.0$ pp):** BN is vital for placket gradient flow |
| **No Augmentation** | Flips and rotations removed | 91.69% | 91.60% | **68.1%** | **Severe drop ($-4.7$ pp):** Memorizes rigid training orientations |
| **2-Block (SimpleCNN)** | 3rd block removed ($7\times7$ pool) | 89.44% | 89.45% | 70.5% | Receptive field insufficient for global silhouette |

### 3.5. The $28 \times 28$ Empirical Resolution Frontier
Even with deep convolutional networks and transfer learning, Shirt recall stabilizes at ${\approx}73.2\%$--$73.4\%$. Three fundamental factors govern this ceiling:
1. **Nyquist-Shannon Sampling Limits:** At $28 \times 28$, a button placket is $1$--$2$ pixels wide. When downsampled, subtle collar tips blend into adjacent pixels, setting an information-theoretic limit.
2. **Annotation Noise Floor:** Garments with hybrid morphology (collarless buttoned henleys, fine-knit crewnecks) exhibit inherent inter-annotator ambiguity in Fashion-MNIST.
3. **Why ImageNet RegNet Did Not Beat DeepCNN:** Pretrained on $224 \times 224$ RGB photographs, RegNet's aggressive initial stride-2 stem downsizes $28 \times 28$ inputs to $14 \times 14$ immediately, causing early spatial information loss.

### 3.6. Production Recommendation: DeepCNN is Pareto-Optimal
For commercial e-commerce inventory tagging, **DeepCNN (3-block)** is the unequivocal deployment choice:
- **Highest Macro $F_1$:** $92.16\%$ (beats all classical and transfer models).
- **High Throughput:** $0.029$ ms per image ($>34{,}000$ images/second per CPU core).
- **Compact Memory Footprint:** $390{,}634$ parameters (${\approx}1.5$ MB storage, $4.4\times$ smaller than Random Forest).

---

## 4. Team Work Breakdown & Individual Contributions

Every component of this project was individually owned, implemented, and verified by a dedicated team member across both Phase 1 and Phase 2:

| Member | GitHub Handle | Phase 1 (AML) Ownership | Phase 2 (DL) Ownership | Key Owned Files |
|---|---|---|---|---|
| **Anant Jain** | [`@anantjain-afk`](https://github.com/anantjain-afk) | **Repository Scaffolding, Data Pipeline, EDA & Master Synthesis** | **PyTorch Dataset Pipeline & Data Augmentation** | `src/aml/data.py`<br>`src/aml/evaluation.py`<br>`src/dl/data.py`<br>`notebooks/01_exploratory_data_analysis.ipynb`<br>`notebooks/00_master_aml_comparison.ipynb` |
| **Kartik Mehra** | [`@KartikMehra22`](https://github.com/KartikMehra22) | **Feature Extraction Engine (Pixels + HOG), Caching Utility & Exploration** | **Deep Learning Training Engine, Optimization & Early Stopping** | `src/aml/features.py`<br>`src/dl/training.py`<br>`notebooks/02_feature_exploration.ipynb`<br>`data/cache/` utility |
| **Krishna Gehlot** | [`@NSTKrishna`](https://github.com/NSTKrishna) | **Model 1: Multinomial Logistic Regression & Weight Spatial Heatmaps** | **Main CNN Training, Benchmarking & Confusion Matrices** | `src/aml/models/logreg.py`<br>`notebooks/03_logistic_regression.ipynb`<br>`notebooks/07_cnn_experiments.ipynb`<br>`phase2_reports/results/cnn_*.json` |
| **Vipul Sharma** | [`@vs-code06`](https://github.com/vs-code06) | **Model 2: $k$-Nearest Neighbors ($k$-NN) & Latency/Scaling Analysis** | **CNN Model Architectures, Unit Testing & 5-Variable Ablation Study** | `src/aml/models/knn.py`<br>`src/dl/models/cnn.py`<br>`tests/test_cnn.py`<br>`notebooks/04_knn_experiments.ipynb`<br>`notebooks/08_ablation_study.ipynb`<br>`phase2_reports/results/ablation_study.json` |
| **Aditya** | [`@AdityaInnovates`](https://github.com/AdityaInnovates) | **Model 3: Random Forest Ensemble, Overfitting & Feature Importance** | **Pretrained RegNet-Y-400MF Transfer Learning & Adaptation** | `src/aml/models/random_forest.py`<br>`src/dl/models/regnet.py`<br>`notebooks/05_random_forest.ipynb`<br>`notebooks/09_regnet_transfer_learning.ipynb`<br>`phase2_reports/results/regnet_finetuned.json` |
| **Utkarsh Jain** | [`@UJ474`](https://github.com/UJ474) | **Model 4: Linear SVM, Error Analysis Toolkit & Gap Documentation** | **Cross-Phase Error Analysis & IEEE LaTeX Technical Report** | `src/aml/models/svm.py`<br>`src/aml/error_analysis.py`<br>`reports/error_analysis.md`<br>`phase2_reports/phase2_error_analysis.md`<br>`phase2_reports/latex_report/main.tex` |

---

## 5. Repository Structure

```
Image-Tagging/
├── README.md                                  # Comprehensive project landing page (Phase 1 + 2)
├── requirements.txt                           # Complete project dependencies (PyTorch + scikit-learn)
├── pyproject.toml                             # Package configuration
├── phase2_plan.md                             # Phase 2 strategic roadmap & milestones
├── phase2_team_guide.md                       # Phase 2 execution guide & team contract
├── docs/team/                                 # Phase 1 team documentation & technical guides
│   ├── README.md                              # Master team coordination protocol
│   ├── 01_Anant_Setup_Data_MasterNotebook.md
│   ├── 02_Kartik_Feature_Extraction_HOG.md
│   ├── 03_Krishna_Logistic_Regression.md
│   ├── 04_Vipul_KNN_Classifier.md
│   ├── 05_Aditya_Random_Forest.md
│   └── 06_Utkarsh_SVM_Error_Analysis.md
├── src/                                       # Core modular source code
│   ├── aml/                                   # Phase 1 Applied Machine Learning package
│   │   ├── data.py                            # Fashion-MNIST loader & stratified 48k/12k split
│   │   ├── features.py                        # Raw Pixels (784) & HOG (324) feature extractors
│   │   ├── evaluation.py                      # Team metric calculation & batch latency timer
│   │   ├── error_analysis.py                  # Confusion matrices & error gallery extractors
│   │   └── models/                            # Classical ML models
│   │       ├── logreg.py                      # Multinomial Logistic Regression pipeline
│   │       ├── knn.py                         # k-NN pipeline with batched matrix inference
│   │       ├── random_forest.py               # Depth-regularized Random Forest ensemble
│   │       └── svm.py                         # Linear Support Vector Machine pipeline
│   └── dl/                                    # Phase 2 Deep Learning package
│       ├── data.py                            # PyTorch Dataset, DataLoaders & Augmentations
│       ├── training.py                        # PyTorch training loop, EarlyStopping & Scheduler
│       └── models/                            # Neural network architectures
│           ├── cnn.py                         # SimpleCNN (2-block) & DeepCNN (3-block)
│           └── regnet.py                      # RegNet-Y-400MF stem adaptation for grayscale
├── notebooks/                                 # Complete end-to-end Jupyter Notebooks
│   ├── 00_master_aml_comparison.ipynb        # Phase 1 Master benchmark table & synthesis
│   ├── 01_exploratory_data_analysis.ipynb     # Phase 1 EDA & class distribution auditing
│   ├── 02_feature_exploration.ipynb           # Phase 1 Pixels vs. HOG visual exploration
│   ├── 03_logistic_regression.ipynb           # Phase 1 LogReg tuning & weight heatmaps
│   ├── 04_knn_experiments.ipynb               # Phase 1 k-NN tuning & latency scaling
│   ├── 05_random_forest.ipynb                 # Phase 1 Random Forest & feature importances
│   ├── 06_svm_and_error_analysis.ipynb        # Phase 1 SVM tuning & initial error analysis
│   ├── 07_cnn_experiments.ipynb               # Phase 2 SimpleCNN & DeepCNN training & evaluation
│   ├── 08_ablation_study.ipynb                # Phase 2 5-Variable controlled ablation study
│   └── 09_regnet_transfer_learning.ipynb      # Phase 2 RegNet transfer learning fine-tuning
├── tests/                                     # Automated test suite (43 unit tests)
│   ├── test_foundation.py                     # Data loader, splits, checksums & evaluation
│   ├── test_features.py                       # Pixel & HOG extraction and caching tests
│   ├── test_logreg.py                         # LogReg pipeline & metric serialization
│   ├── test_knn.py                            # k-NN pipeline, batching & distance tests
│   ├── test_svm.py                            # SVM pipeline, tuning & error helpers
│   ├── test_master_notebook.py                # Master comparison table integrity tests
│   └── test_cnn.py                            # PyTorch forward/backward pass & ablation tests
├── reports/                                   # Phase 1 artifacts & publications
│   ├── error_analysis.md                      # Phase 1 error report defining "The Gap"
│   ├── technical_report.tex                   # Phase 1 IEEE Conference Paper (LaTeX)
│   ├── technical_report.pdf                   # Compiled Phase 1 Technical Report
│   ├── figures/                               # Phase 1 exported figures (EDA, features, models)
│   └── results/                               # 8 Phase 1 standardized JSON metric records
└── phase2_reports/                            # Phase 2 artifacts & publications
    ├── phase2_error_analysis.md               # Phase 2 comprehensive error analysis report
    ├── latex_report/                          # Phase 2 IEEE Technical Report (Conference Paper)
    │   ├── main.tex                           # Overleaf-ready LaTeX source with full tables
    │   ├── references.bib                     # Verified BibTeX bibliography
    │   └── figures/                           # Bundled high-resolution figures
    ├── latex_report.zip                       # 1-Click ready-to-upload Overleaf archive
    ├── figures/                               # Phase 2 training curves & confusion comparisons
    └── results/                               # Standardized Phase 2 JSON metric records
        ├── cnn_simple.json                    # SimpleCNN evaluation results
        ├── cnn_deep.json                      # DeepCNN evaluation results
        ├── regnet_finetuned.json              # Fine-tuned RegNet evaluation results
        └── ablation_study.json                # 5-condition ablation records
```

---

## 6. Installation & Environment Setup

### 6.1. Clone Repository & Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/anantjain-afk/Image-Tagging.git
cd Image-Tagging

# Create virtual environment with Python 3.10+
python3 -m venv .venv
source .venv/bin/activate  # On Windows PowerShell: .\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

### 6.2. Verify Environment & PyTorch Setup

```bash
python -c "import torch, torchvision, sklearn; print('PyTorch:', torch.__version__, '| Torchvision:', torchvision.__version__, '| Device:', 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')"
```

---

## 7. How to Run Everything

### 7.1. Run the Automated Unit Test Suite (43 Tests)
To verify data pipeline integrity, feature shapes, metric serialization, and PyTorch model gradient flows:

```bash
python -m unittest discover -s tests -v
```
*Expected: 43 tests pass cleanly with `OK`.*

### 7.2. Verify Data Pipeline & Precompute Feature Cache
Download official Fashion-MNIST binaries with automatic checksum validation and precompute HOG/pixel caches:

```bash
# Check splits: 48,000 train / 12,000 val / 10,000 test
python -c "from aml.data import get_train_val_test_splits; d = get_train_val_test_splits(); print('Splits OK:', [v.shape for k, v in d.items() if 'X_' in k])"

# Precompute offline feature cache (saves time across notebooks)
python src/aml/features.py
```

### 7.3. Run Phase 1 AML Experiments
Open and execute the individual Jupyter notebooks in order:
```bash
jupyter lab notebooks/01_exploratory_data_analysis.ipynb
jupyter lab notebooks/02_feature_exploration.ipynb
jupyter lab notebooks/03_logistic_regression.ipynb
jupyter lab notebooks/04_knn_experiments.ipynb
jupyter lab notebooks/05_random_forest.ipynb
jupyter lab notebooks/06_svm_and_error_analysis.ipynb
jupyter lab notebooks/00_master_aml_comparison.ipynb
```

### 7.4. Run Phase 2 Deep Learning Experiments
```bash
# Model Benchmarks: Train SimpleCNN and DeepCNN
jupyter lab notebooks/07_cnn_experiments.ipynb

# Controlled 5-Variable Ablation Study
jupyter lab notebooks/08_ablation_study.ipynb

# Hybrid Innovation: RegNet-Y-400MF Transfer Learning
jupyter lab notebooks/09_regnet_transfer_learning.ipynb
```

### 7.5. Generate Phase 2 Figures & Error Visualizations
```bash
python phase2_reports/figures/gen_figures.py
```
*Generates `aml_vs_cnn_confusion.png` and `shirt_recovery_gallery.png` in `phase2_reports/figures/`.*

### 7.6. Compile the IEEE LaTeX Technical Report
The complete IEEE conference paper comparing Phase 1 and Phase 2 is packaged in `phase2_reports/latex_report/`:

- **Option A (Overleaf — Recommended):**
  1. Go to [Overleaf](https://www.overleaf.com) $\to$ **New Project** $\to$ **Upload Project**.
  2. Upload [`phase2_reports/latex_report.zip`](phase2_reports/latex_report.zip).
  3. Ensure compiler is set to **pdfLaTeX** and main document is **`main.tex`**.
  4. Click **Recompile**.
- **Option B (Local TeX Live):**
  ```bash
  cd phase2_reports/latex_report
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex
  ```

---

## 8. License & Attribution

This project was developed for **Course Project 11, Section D** at our institution.  
- **Dataset:** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) is provided by Zalando Research under the [MIT License](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE).
- **Codebase:** Distributed under the MIT License. See individual files for licensing notices.
