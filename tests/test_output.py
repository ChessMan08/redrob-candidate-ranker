"""
tests/test_output.py — Tests for submission CSV writing and validation.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.utils.output import write_submission, validate_submission_locally
from src.scoring.composite import CandidateScore


def _make_dummy_scores(n: int = 100) -> list[CandidateScore]:
    """Create n dummy CandidateScore objects with decreasing composites."""
    scores = []
    for i in range(n):
        c = {
            "candidate_id": f"CAND_{i+1:07d}",
            "profile": {
                "current_title": "ML Engineer",
                "current_company": "Swiggy",
                "current_industry": "food delivery",
                "years_of_experience": 6.0,
                "location": "Hyderabad",
                "country": "India",
                "headline": "Test",
                "summary": "Test summary",
            },
            "career_history": [{
                "company": "Swiggy",
                "title": "ML Engineer",
                "duration_months": 36,
                "is_current": True,
                "industry": "food delivery",
                "company_size": "1001-5000",
                "description": "Built search ranking with FAISS and embeddings.",
                "start_date": None,
                "end_date": None,
            }],
            "skills": [],
            "education": [],
            "redrob_signals": {
                "last_active_date": None,
                "open_to_work_flag": True,
                "recruiter_response_rate": 0.8,
                "notice_period_days": 30,
                "github_activity_score": 50.0,
                "saved_by_recruiters_30d": 5,
                "interview_completion_rate": 0.9,
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True,
                "applications_submitted_30d": 2,
                "willing_to_relocate": True,
                "skill_assessment_scores": {},
                "avg_response_time_hours": 12.0,
                "profile_completeness_score": 80.0,
                "salary_min_lpa": 25.0,
                "salary_max_lpa": 40.0,
            },
        }
        cs = CandidateScore(
            candidate_id=f"CAND_{i+1:07d}",
            composite=round(100.0 - i * 0.9, 6),
            career_score=80.0,
            skills_score=70.0,
            experience_score=100.0,
            behavioral_score=75.0,
            location_score=100.0,
            education_score=75.0,
            tier1_skills=["FAISS", "Pinecone"],
            tier2_skills=["PyTorch", "NLP"],
            candidate=c,
        )
        scores.append(cs)
    return scores


class TestWriteSubmission:

    def test_writes_correct_number_of_rows(self, tmp_path):
        scored = _make_dummy_scores(150)
        out = tmp_path / "submission.csv"
        write_submission(scored, out, top_n=100)
        errors = validate_submission_locally(out)
        assert errors == [], f"Validation errors: {errors}"

    def test_scores_monotonically_non_increasing(self, tmp_path):
        import csv
        scored = _make_dummy_scores(120)
        out = tmp_path / "sub.csv"
        write_submission(scored, out, top_n=100)
        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        scores = [float(r["score"]) for r in rows]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], \
                f"Score dropped at rank {i+1}: {scores[i]} -> {scores[i+1]}"

    def test_all_ranks_present(self, tmp_path):
        import csv
        scored = _make_dummy_scores(120)
        out = tmp_path / "sub.csv"
        write_submission(scored, out, top_n=100)
        with open(out) as f:
            reader = csv.DictReader(f)
            ranks = {int(r["rank"]) for r in reader}
        assert ranks == set(range(1, 101))

    def test_no_duplicate_candidate_ids(self, tmp_path):
        import csv
        scored = _make_dummy_scores(120)
        out = tmp_path / "sub.csv"
        write_submission(scored, out, top_n=100)
        with open(out) as f:
            reader = csv.DictReader(f)
            ids = [r["candidate_id"] for r in reader]
        assert len(ids) == len(set(ids))

    def test_reasoning_is_non_empty(self, tmp_path):
        import csv
        scored = _make_dummy_scores(110)
        out = tmp_path / "sub.csv"
        write_submission(scored, out, top_n=100)
        with open(out) as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert row["reasoning"].strip(), \
                    f"Empty reasoning for {row['candidate_id']}"


class TestValidateLocally:

    def test_valid_csv_passes(self, tmp_path):
        scored = _make_dummy_scores(110)
        out = tmp_path / "team_x.csv"
        write_submission(scored, out)
        errors = validate_submission_locally(out)
        assert errors == []

    def test_wrong_extension_fails(self, tmp_path):
        out = tmp_path / "submission.xlsx"
        out.write_text("dummy")
        errors = validate_submission_locally(out)
        assert any("csv" in e.lower() for e in errors)

    def test_wrong_row_count_fails(self, tmp_path):
        import csv
        out = tmp_path / "sub.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["candidate_id", "rank", "score", "reasoning"])
            for i in range(50):  # only 50 instead of 100
                w.writerow([f"CAND_{i+1:07d}", i+1, round(100 - i, 2), "test reason"])
        errors = validate_submission_locally(out)
        assert any("100" in e for e in errors)

    def test_duplicate_rank_fails(self, tmp_path):
        import csv
        scored = _make_dummy_scores(110)
        out = tmp_path / "sub.csv"
        # Write manually with duplicate rank
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["candidate_id", "rank", "score", "reasoning"])
            for i in range(100):
                rank = 1 if i < 2 else i + 1  # duplicate rank 1
                w.writerow([f"CAND_{i+1:07d}", rank, round(100 - i * 0.9, 6), "reason"])
        errors = validate_submission_locally(out)
        assert any("duplicate" in e.lower() or "rank" in e.lower() for e in errors)
