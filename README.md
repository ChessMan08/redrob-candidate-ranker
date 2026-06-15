# Redrob Candidate Ranker

**Intelligent Candidate Discovery & Ranking — Hackathon Submission**

A production-grade candidate ranking system that scores 100,000 candidates for role fit using a multi-signal structured scorer with TF-IDF re-ranking. Runs in **< 3 minutes** on CPU with 16 GB RAM, no GPU, no network access during inference.

---

## Folder Structure

```
redrob_ranker/
├── rank.py                        # Main inference entry point
├── requirements.txt
├── app.py
├── requirements-streamlit.txt
├── README.md
│
├── src/
│   ├── config/
│   │   └── settings.py            # All weights, keyword lists, thresholds
│   │
│   ├── data/
│   │   ├── loader.py              # Streaming JSONL/GZ loader + deduplication
│   │   └── preprocessor.py        # Cleaning, normalisation, type coercion
│   │
│   ├── features/
│   │   ├── career_features.py     # Career trajectory scorer (company+title+industry)
│   │   ├── skill_features.py      # Credibility-gated skill scorer
│   │   ├── behavioral_features.py # Availability + engagement scorer
│   │   ├── profile_features.py    # Experience, location, education
│   │   └── honeypot_detector.py   # Detects impossible/inconsistent profiles
│   │
│   ├── scoring/
│   │   ├── composite.py           # Assembles all features → CandidateScore
│   │   └── reasoning.py           # Generates grounded per-candidate reasoning
│   │
│   ├── retrieval/
│   │   ├── tfidf_reranker.py      # TF-IDF re-ranking of top-500
│   │   └── bm25_filter.py         # Optional BM25 pre-filter
│   │
│   ├── evaluation/
│   │   ├── metrics.py             # NDCG@K, MAP, P@K, MRR implementations
│   │   └── evaluator.py           # Full evaluation report builder
│   │
│   └── utils/
│       ├── output.py              # CSV writer + local validator
│       ├── parallel.py            # Multi-process batch scorer
│       └── inspect_data.py        # EDA and data quality reporting
│
├── scripts/
│   ├── inspect.py                 # Run EDA on candidate dataset
│   ├── evaluate.py                # Offline evaluation with sensitivity analysis
│   ├── annotate.py                # Interactive manual labelling CLI
│   └── tune_weights.py            # Grid-search weight tuning
│
├── tests/
│   ├── test_features.py           # Unit tests for all feature scorers
│   ├── test_output.py             # Tests for CSV writing and validation
│   └── test_metrics.py            # Tests for ranking evaluation metrics
│
└── artifacts/
    ├── manual_labels.json         # Hand-annotated relevance grades (50 sample)
    └── top100_detail.json         # Generated: detailed breakdown for top-100
```

---

## Architecture

### Why not embeddings / LLMs?

This challenge has three hard constraints that rule out naïve approaches:

1. **Compute**: 5 minutes, CPU only, no network → rules out per-candidate LLM calls and embedding models during inference
2. **Honeypot traps**: ~80 impossible profiles designed to trick keyword/embedding systems
3. **Narrow JD**: The ideal candidate is explicitly described — this is specification execution, not general semantic matching

### Solution: Structured Scorer + TF-IDF Re-rank

```
Load & Deduplicate
      ↓
Clean & Normalise (fix inverted salaries, parse dates, gate missing fields)
      ↓
Score each candidate:
  ├─ Career Trajectory (35%)  — company type + title type + industry
  │                              + title-coherence penalty for non-ML titles
  │                              + description ML keyword signal
  ├─ Skills (30%)             — credibility-gated (duration × endorsements × assessment)
  │                              Tier-1 (retrieval/ranking) >> Tier-2 (ML) >> Tier-3
  ├─ Experience (13%)         — YoE curve: sweet-spot 6-8yr = 100
  ├─ Behavioral (12%)         — availability × engagement (60/40 blend)
  ├─ Location (6%)            — India preferred; Pune/Noida/Hyd/Blr ideal
  └─ Education (4%)           — tier_1/2 CS is a soft bonus
      ↓
Apply multipliers:
  - Honeypot detector        (0.20 – 1.0)
  - Behavioral hard gate     (0.50 – 1.0) for extreme unavailability
  - Salary fit               (0.90 – 1.0) mild
  - Skills minimum gate      (cap at 45 if skills_score < 5)
      ↓
TF-IDF re-rank top-500       (15% weight, JD intent representation)
      ↓
Sort → top-100 → generate reasoning → write CSV
```

---

## Installation

```bash
# Python 3.9+
pip install -r requirements.txt
```

**Core dependencies** (all CPU-only, offline-safe):
```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0    # TF-IDF only
python-dateutil>=2.8.2
pyyaml>=6.0
rank-bm25>=0.2.2       # optional BM25 pre-filter
tqdm>=4.65.0           # optional progress bars
```

---

## Quick Start

### 1. Inspect the dataset

```bash
python scripts/inspect.py --candidates sample_candidates.json
python scripts/inspect.py --candidates candidates.jsonl.gz --max 5000
```

### 2. Run inference (main command)

```bash
# Basic run — structured scorer only
python rank.py --candidates candidates.jsonl.gz --out submission.csv

# With TF-IDF re-ranking (recommended, adds ~10s)
python rank.py --candidates candidates.jsonl.gz --out submission.csv --tfidf

# Parallel scoring (faster for 100K candidates)
python rank.py --candidates candidates.jsonl.gz --out submission.csv --tfidf --workers 8

# Skip evaluation report for max speed
python rank.py --candidates candidates.jsonl.gz --out submission.csv --tfidf --workers 8 --no-eval
```

**Expected runtime**: ~60-90s (structured only) or ~75-100s (with TF-IDF) on 8-core CPU, 16 GB RAM.

### 3. Validate submission

```bash
# Internal validator (same checks as official)
python -c "
from src.utils.output import validate_submission_locally
errors = validate_submission_locally('submission.csv')
print('VALID' if not errors else errors)
"

# Official validator
python validate_submission.py submission.csv
```

### 4. Evaluate offline

```bash
# Full evaluation report
python scripts/evaluate.py --candidates candidates.jsonl.gz --tfidf --save-scores

# With sensitivity analysis
python scripts/evaluate.py --candidates candidates.jsonl.gz --sensitivity

# Quick check on sample
python scripts/evaluate.py --candidates sample_candidates.json --tfidf
```

### 5. Manual annotation (for ground truth)

```bash
# Annotate top-100 candidates interactively
python scripts/annotate.py --candidates candidates.jsonl.gz --top 100

# After annotating, run evaluation with manual labels
python scripts/evaluate.py --candidates candidates.jsonl.gz
```

### 6. Weight tuning

```bash
# Grid-search weight configurations
python scripts/tune_weights.py --candidates candidates.jsonl.gz

# Quick tune on sample
python scripts/tune_weights.py --candidates sample_candidates.json
```

### 7. Run tests

```bash
# With pytest
pytest tests/ -v

# Without pytest (stdlib)
python -m unittest discover tests/
```

---

## Tuning

All weights and thresholds are in `src/config/settings.py`. The most impactful levers:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `WEIGHTS["career"]` | 0.35 | Increase if IT-services candidates still ranking too high |
| `WEIGHTS["skills"]` | 0.30 | Increase if good ML engineers are being missed |
| `WEIGHTS["behavioral"]` | 0.12 | Decrease if available but non-ML candidates rank too high |
| `TFIDF_BLEND_WEIGHT` | 0.15 | Increase to 0.20 if TF-IDF is finding more relevant candidates |
| Skills gate cap | 45.0 | Lower to 40 to push non-ML candidates further down |

After changing weights, run:
```bash
python scripts/evaluate.py --candidates sample_candidates.json
python scripts/tune_weights.py --candidates candidates.jsonl.gz --max 5000
```

---

## Key Design Decisions

### Honeypot Defense
Skills with `duration_months == 0` AND `endorsements == 0` are silently ignored. The `detect_honeypot()` function penalises (but never hard-excludes) profiles with:
- High endorsements + zero duration (endorsement fraud)
- Expert self-rating + poor Redrob assessment score (<25/100)
- 8+ expert skills with < 50% having any supporting evidence

### IT Services Penalty
Candidates whose entire career is at IT services firms (TCS, Infosys, Wipro, etc.) get a 40% career-score reduction. Each individual IT services role gets −30 points. This is the JD's most explicit requirement: "not a services-company background."

### Title-Coherence Penalty
Frontend Engineers, Java Developers, QA Engineers, and DevOps Engineers at product companies get a 25% career score reduction if they have no ML title history. Having OpenSearch or FAISS in their skills (even with real usage) doesn't make a Frontend Engineer an ML candidate.

### Behavioral Gating
Candidates inactive > 1 year + not open-to-work + RRR < 10% get their composite score multiplied by 0.50. This is a hard-floor multiplier, not a soft penalty — an unreachable candidate is not a useful shortlist entry.

---

## Assumptions (Unseen Full Dataset)

1. **Distribution**: ~5-10% of 100K candidates will have some ML/retrieval skills; ~1-2% will be genuinely strong fits.
2. **Salary data**: ~30% of records will have inverted min/max — the preprocessor swaps them silently.
3. **Company names**: Some company names will be variants not in our lists (e.g. "Google India Pvt Ltd"). The company-size heuristic (51-200 employees → likely product startup) catches unknowns.
4. **Honeypots**: ~80 of 100K have impossible profiles. Our honeypot detector will flag them with multipliers of 0.2-0.7; they will not reach the top 100.
5. **Duplicate candidates**: A small fraction may have duplicate candidate_ids. The deduplicator keeps the first occurrence.
6. **Behavioral dates**: `last_active_date` is relative to the dataset creation date. We use Python's `date.today()` — if the dataset is old, all candidates will appear inactive. Adjust `src/config/settings.py: TODAY` if needed.
