import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from datetime import date
from unittest.mock import patch

from src.features.career_features import (
    score_career,
    has_ever_worked_at_product_company,
    years_in_ml_roles,
)
from src.features.skill_features import score_skills, top_tier1_skills
from src.features.behavioral_features import (
    score_behavioral,
    behavioral_multiplier,
)
from src.features.profile_features import (
    score_experience,
    score_location,
    score_education,
)
from src.features.honeypot_detector import detect_honeypot
from src.data.preprocessor import clean_candidate

# Fixtures
def make_role(company="Swiggy", title="ML Engineer", industry="food delivery",
              company_size="1001-5000", duration=24, is_current=True,
              description="Built ranking and recommendation systems using FAISS and embeddings."):
    return {
        "company": company,
        "title": title,
        "industry": industry,
        "company_size": company_size,
        "duration_months": duration,
        "is_current": is_current,
        "description": description,
        "start_date": date(2022, 1, 1),
        "end_date": None,
    }


def make_skill(name, proficiency="advanced", endorsements=20, duration_months=18):
    return {
        "name": name.lower(),
        "name_raw": name,
        "proficiency": proficiency,
        "endorsements": endorsements,
        "duration_months": duration_months,
    }


def make_signal(**kwargs):
    defaults = {
        "profile_completeness_score": 80.0,
        "last_active_date": date.today(),
        "open_to_work_flag": True,
        "recruiter_response_rate": 0.8,
        "avg_response_time_hours": 12.0,
        "notice_period_days": 30,
        "github_activity_score": 60.0,
        "saved_by_recruiters_30d": 5,
        "interview_completion_rate": 0.9,
        "offer_acceptance_rate": 0.7,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
        "applications_submitted_30d": 3,
        "willing_to_relocate": True,
        "skill_assessment_scores": {},
        "salary_min_lpa": 25.0,
        "salary_max_lpa": 40.0,
        "preferred_work_mode": "hybrid",
        "signup_date": date(2023, 1, 1),
        "profile_views_received_30d": 10,
        "connection_count": 500,
        "endorsements_received": 150,
        "search_appearance_30d": 20,
    }
    defaults.update(kwargs)
    return defaults

# Career feature tests
class TestCareerFeatures:

    def test_product_company_ml_title_scores_high(self):
        career = [make_role("Swiggy", "ML Engineer", "food delivery", "1001-5000", 36)]
        score, _ = score_career(career)
        assert score >= 75, f"Product company ML engineer should score >=75, got {score}"

    def test_it_services_scores_low(self):
        career = [make_role("TCS", "Software Engineer", "it services", "10001+", 36)]
        score, _ = score_career(career)
        assert score <= 35, f"IT services candidate should score <=35, got {score}"

    def test_pure_it_services_career_gets_extra_penalty(self):
        career = [
            make_role("TCS", "Software Engineer", "it services", "10001+", 24, False),
            make_role("Infosys", "Senior Engineer", "it services", "10001+", 36, True),
        ]
        score, _ = score_career(career)
        assert score <= 25, f"All-IT-services career should be <=25, got {score}"

    def test_non_ic_title_penalised(self):
        career = [make_role("Swiggy", "Operations Manager", "food delivery", "1001-5000", 24)]
        score, _ = score_career(career)
        # Should be lower than ML engineer at same company
        ml_career = [make_role("Swiggy", "ML Engineer", "food delivery", "1001-5000", 24)]
        ml_score, _ = score_career(ml_career)
        assert score < ml_score, "Ops manager should score lower than ML engineer"

    def test_empty_career_returns_low_score(self):
        score, _ = score_career([])
        assert score == 20.0

    def test_has_ever_worked_at_product_company(self):
        career = [
            make_role("TCS", "Engineer", "it services", "10001+", 24, False),
            make_role("Swiggy", "ML Engineer", "food delivery", "1001-5000", 12, True),
        ]
        assert has_ever_worked_at_product_company(career) is True

    def test_never_worked_at_product_company(self):
        career = [
            make_role("TCS", "Engineer", "it services", "10001+", 24, False),
            make_role("Infosys", "Sr Engineer", "it services", "10001+", 36, True),
        ]
        assert has_ever_worked_at_product_company(career) is False

    def test_description_ml_signal_boosts_score(self):
        career_with_desc = [make_role(
            "Startup Inc", "Engineer", "saas", "51-200", 24,
            description="Built semantic search with FAISS, sentence transformers, and vector databases."
        )]
        career_no_desc = [make_role(
            "Startup Inc", "Engineer", "saas", "51-200", 24,
            description="Managed team projects and delivery timelines."
        )]
        score_with, _ = score_career(career_with_desc)
        score_no, _   = score_career(career_no_desc)
        assert score_with > score_no, "ML keywords in description should boost score"

# Skill feature tests
class TestSkillFeatures:

    def test_tier1_skills_score_high(self):
        # 3 credentialed tier-1 skills; each expert+full-cred = ~12-15 pts → ~37 total
        # 6 skills reaches 60.  Threshold here is "clearly above generic profile".
        skills = [
            make_skill("FAISS", "expert", 30, 24),
            make_skill("Pinecone", "advanced", 15, 18),
            make_skill("sentence-transformers", "advanced", 20, 24),
        ]
        score, _ = score_skills(skills, {})
        assert score >= 20, f"Tier-1 skills should score >=20, got {score}"

    def test_uncredentialed_skills_ignored(self):
        """Skills with 0 duration AND 0 endorsements should not be counted."""
        skills_with = [make_skill("FAISS", "expert", 20, 12)]
        skills_zero = [make_skill("FAISS", "expert", 0, 0)]
        score_with, _ = score_skills(skills_with, {})
        score_zero, _ = score_skills(skills_zero, {})
        assert score_zero == 0.0, f"Uncredentialed skill should score 0, got {score_zero}"
        assert score_with > 0.0

    def test_anti_skill_dominance_penalises(self):
        """Profile dominated by non-ML skills should score low."""
        skills = [
            make_skill("Photoshop", "expert", 30, 36),
            make_skill("Figma", "expert", 25, 30),
            make_skill("Marketing", "advanced", 20, 24),
            make_skill("SEO", "advanced", 15, 18),
            make_skill("Content Writing", "advanced", 12, 12),
        ]
        score, bd = score_skills(skills, {})
        assert score < 15, f"Anti-skill dominated profile should score <15, got {score}"

    def test_assessment_boost(self):
        """High assessment score should boost a skill's contribution."""
        skills = [make_skill("FAISS", "intermediate", 5, 2)]
        score_no_assess, _ = score_skills(skills, {})
        score_with_assess, _ = score_skills(skills, {"FAISS": 90.0})
        assert score_with_assess > score_no_assess

    def test_assessment_penalty_for_expert_low_score(self):
        """Expert claim but poor assessment should reduce contribution."""
        skills = [make_skill("Python NLP", "expert", 20, 24)]
        score_no_assess, _ = score_skills(skills, {})
        score_poor_assess, _ = score_skills(skills, {"Python NLP": 15.0})
        assert score_poor_assess < score_no_assess

    def test_top_tier1_returns_ranked_list(self):
        skills = [
            make_skill("FAISS", "expert", 30, 24),
            make_skill("Pinecone", "advanced", 20, 18),
            make_skill("Python", "advanced", 40, 60),   # Tier-2
        ]
        t1 = top_tier1_skills(skills, {})
        assert "FAISS" in t1
        assert "Python" not in t1   # Python is Tier-2, not Tier-1

# Behavioral feature tests
class TestBehavioralFeatures:

    def test_active_open_high_rrr_scores_high(self):
        sig = make_signal(
            open_to_work_flag=True,
            last_active_date=date.today(),
            recruiter_response_rate=0.9,
            notice_period_days=15,
            github_activity_score=70.0,
        )
        score, _ = score_behavioral(sig)
        assert score >= 75, f"Highly available candidate should score >=75, got {score}"

    def test_inactive_closed_low_rrr_scores_low(self):
        from datetime import timedelta
        sig = make_signal(
            open_to_work_flag=False,
            last_active_date=date.today() - timedelta(days=400),
            recruiter_response_rate=0.03,
            notice_period_days=120,
            github_activity_score=-1.0,
            saved_by_recruiters_30d=0,
            interview_completion_rate=0.5,
            verified_email=False,
            verified_phone=False,
            linkedin_connected=False,
            applications_submitted_30d=0,
            avg_response_time_hours=200.0,
            profile_completeness_score=40.0,
        )
        score, _ = score_behavioral(sig)
        assert score <= 40, f"Unavailable candidate should score <=40, got {score}"

    def test_behavioral_multiplier_normal_is_1(self):
        sig = make_signal()
        mult = behavioral_multiplier(sig)
        assert mult == 1.0

    def test_behavioral_multiplier_dark_candidate(self):
        from datetime import timedelta
        sig = make_signal(
            open_to_work_flag=False,
            last_active_date=date.today() - timedelta(days=500),
            recruiter_response_rate=0.03,
        )
        mult = behavioral_multiplier(sig)
        assert mult <= 0.5, f"Dark candidate multiplier should be <=0.5, got {mult}"

# Profile feature tests
class TestProfileFeatures:

    def test_experience_sweet_spot(self):
        score, _ = score_experience(7.0)
        assert score == 100.0

    def test_experience_too_junior(self):
        score, _ = score_experience(1.5)
        assert score <= 28.0

    def test_experience_very_senior(self):
        score, _ = score_experience(15.0)
        assert score <= 43.0

    def test_location_preferred_city(self):
        profile = {"location": "hyderabad", "country": "india"}
        sig = make_signal(willing_to_relocate=False)
        score, _ = score_location(profile, sig)
        assert score == 100.0

    def test_location_india_wrong_city_will_relocate(self):
        profile = {"location": "jaipur", "country": "india"}
        sig = make_signal(willing_to_relocate=True)
        score, _ = score_location(profile, sig)
        assert 65 <= score <= 80

    def test_location_outside_india_no_reloc(self):
        profile = {"location": "london", "country": "uk"}
        sig = make_signal(willing_to_relocate=False)
        score, _ = score_location(profile, sig)
        assert score <= 20.0

    def test_education_tier1_cs_scores_high(self):
        edu = [{"tier": "tier_1", "field_of_study": "computer science",
                "degree": "b.tech", "start_year": 2015, "end_year": 2019}]
        score, _ = score_education(edu)
        assert score >= 100.0

    def test_education_no_edu_returns_neutral(self):
        score, _ = score_education([])
        assert score == 50.0

# Honeypot detector tests
class TestHoneypotDetector:

    def _make_candidate(self, skills, career=None, yoe=6.0):
        return {
            "candidate_id": "CAND_TEST001",
            "profile": {"years_of_experience": yoe},
            "career_history": career or [make_role()],
            "skills": skills,
            "redrob_signals": {"skill_assessment_scores": {}},
        }

    def test_clean_profile_multiplier_is_1(self):
        skills = [make_skill("FAISS", "advanced", 15, 18)]
        c = self._make_candidate(skills)
        mult, flags = detect_honeypot(c)
        assert mult == 1.0
        assert len(flags) == 0

    def test_high_endorse_zero_duration_flagged(self):
        skills = [make_skill("FAISS", "advanced", 50, 0)]  # 50 endorsements, 0 months
        c = self._make_candidate(skills)
        mult, flags = detect_honeypot(c)
        assert mult < 1.0
        assert any("high_endorse_zero_dur" in f for f in flags)

    def test_expert_low_assessment_flagged(self):
        skills = [make_skill("NLP", "expert", 10, 12)]
        c = self._make_candidate(skills)
        c["redrob_signals"]["skill_assessment_scores"] = {"NLP": 10.0}
        mult, flags = detect_honeypot(c)
        assert mult < 1.0
        assert any("expert_low_assess" in f for f in flags)

    def test_mass_expert_claims_flagged(self):
        # 10 expert skills, most with zero evidence
        skills = [
            make_skill(f"skill_{i}", "expert", 0, 0)
            for i in range(10)
        ]
        c = self._make_candidate(skills)
        mult, flags = detect_honeypot(c)
        assert mult < 1.0

    def test_multiplier_floor_is_0_2(self):
        """Even the worst honeypot gets at least 0.2 multiplier (no hard exclusion)."""
        skills = [
            make_skill(f"s{i}", "expert", 50, 0)  # mass endorsement fraud
            for i in range(12)
        ]
        c = self._make_candidate(skills)
        c["redrob_signals"]["skill_assessment_scores"] = {f"s{i}": 5.0 for i in range(12)}
        mult, _ = detect_honeypot(c)
        assert mult >= 0.20

# Integration test: preprocessor + scorer
class TestEndToEnd:

    def _raw_candidate(self, cid="CAND_0000001"):
        return {
            "candidate_id": cid,
            "profile": {
                "anonymized_name": "Test Candidate",
                "headline": "ML Engineer specializing in search and ranking",
                "summary": "Built production ranking systems with FAISS and Pinecone.",
                "location": "Hyderabad",
                "country": "India",
                "years_of_experience": 7.0,
                "current_title": "ML Engineer",
                "current_company": "Swiggy",
                "current_company_size": "1001-5000",
                "current_industry": "Food Delivery",
            },
            "career_history": [{
                "company": "Swiggy",
                "title": "ML Engineer",
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": 54,
                "is_current": True,
                "industry": "food delivery",
                "company_size": "1001-5000",
                "description": "Built recommendation and search ranking systems using FAISS, "
                               "Pinecone, sentence-transformers. Measured with NDCG and MRR.",
            }],
            "education": [{
                "institution": "IIT Hyderabad",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2013,
                "end_year": 2017,
                "tier": "tier_1",
                "grade": "8.5",
            }],
            "skills": [
                {"name": "FAISS", "proficiency": "advanced", "endorsements": 25, "duration_months": 30},
                {"name": "Pinecone", "proficiency": "advanced", "endorsements": 18, "duration_months": 24},
                {"name": "sentence-transformers", "proficiency": "advanced", "endorsements": 20, "duration_months": 24},
                {"name": "Python", "proficiency": "expert", "endorsements": 50, "duration_months": 60},
                {"name": "PyTorch", "proficiency": "advanced", "endorsements": 22, "duration_months": 30},
            ],
            "certifications": [],
            "languages": [{"language": "English", "proficiency": "native"}],
            "redrob_signals": {
                "profile_completeness_score": 92.0,
                "signup_date": "2023-01-15",
                "last_active_date": str(date.today()),
                "open_to_work_flag": True,
                "profile_views_received_30d": 20,
                "applications_submitted_30d": 5,
                "recruiter_response_rate": 0.85,
                "avg_response_time_hours": 8.0,
                "skill_assessment_scores": {"FAISS": 88.0, "Python": 92.0},
                "connection_count": 600,
                "endorsements_received": 250,
                "notice_period_days": 30,
                "expected_salary_range_inr_lpa": {"min": 30.0, "max": 45.0},
                "preferred_work_mode": "hybrid",
                "willing_to_relocate": True,
                "github_activity_score": 72.0,
                "search_appearance_30d": 35,
                "saved_by_recruiters_30d": 12,
                "interview_completion_rate": 0.95,
                "offer_acceptance_rate": 0.8,
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True,
            },
        }

    def test_ideal_candidate_scores_high(self):
        from src.data.preprocessor import clean_candidate
        from src.scoring.composite import score_candidate
        cleaned = clean_candidate(self._raw_candidate())
        cs = score_candidate(cleaned)
        assert cs.composite >= 75.0, f"Ideal candidate should score >=75, got {cs.composite}"

    def test_salary_inversion_handled_gracefully(self):
        """Inverted salary (min > max) should not crash the scorer."""
        from src.data.preprocessor import clean_candidate
        from src.scoring.composite import score_candidate
        raw = self._raw_candidate()
        raw["redrob_signals"]["expected_salary_range_inr_lpa"] = {"min": 50.0, "max": 20.0}
        cleaned = clean_candidate(raw)
        cs = score_candidate(cleaned)
        assert cs is not None

    def test_missing_fields_handled(self):
        """Missing optional fields should not crash."""
        from src.data.preprocessor import clean_candidate
        from src.scoring.composite import score_candidate
        raw = self._raw_candidate()
        del raw["education"]
        del raw["certifications"]
        del raw["languages"]
        cleaned = clean_candidate(raw)
        cs = score_candidate(cleaned)
        assert cs is not None
        assert cs.composite > 0

    def test_reasoning_is_non_empty(self):
        from src.data.preprocessor import clean_candidate
        from src.scoring.composite import score_candidate
        from src.scoring.reasoning import generate_reasoning
        cleaned = clean_candidate(self._raw_candidate())
        cs      = score_candidate(cleaned)
        reason  = generate_reasoning(cs, rank=1)
        assert isinstance(reason, str)
        assert len(reason) > 20

    def test_score_monotone_for_sorted_list(self):
        """After sorting, scores must be monotonically non-increasing."""
        from src.data.preprocessor import clean_candidate
        from src.scoring.composite import score_candidates
        raws = [self._raw_candidate(f"CAND_{i:07d}") for i in range(1, 6)]
        # Vary one param to get different scores
        raws[1]["profile"]["years_of_experience"] = 2.0
        raws[2]["redrob_signals"]["recruiter_response_rate"] = 0.01
        raws[3]["career_history"][0]["company"] = "TCS"
        raws[3]["career_history"][0]["industry"] = "it services"
        cleaned = [clean_candidate(r) for r in raws]
        scored = score_candidates(cleaned)
        for i in range(len(scored) - 1):
            assert scored[i].composite >= scored[i + 1].composite, \
                f"Score at rank {i+1} ({scored[i].composite}) < rank {i+2} ({scored[i+1].composite})"
