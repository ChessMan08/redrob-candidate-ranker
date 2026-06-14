"""
settings.py — Central configuration for the Redrob candidate ranker.

All numeric weights, keyword lists, and thresholds live here so the
rest of the code stays clean.  Change weights here; re-run rank.py.
"""

from datetime import date
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Runtime date (used for recency calculations)
# ─────────────────────────────────────────────────────────────────────────────
TODAY: date = date.today()

# ─────────────────────────────────────────────────────────────────────────────
# Composite score weights  (must sum to 1.0)
# ─────────────────────────────────────────────────────────────────────────────
WEIGHTS = {
    "career":       0.35,   # company type + title type — primary discriminator
    "skills":       0.30,   # ML/retrieval skills with credibility gating
    "experience":   0.13,   # YoE in the sweet-spot band
    "behavioral":   0.12,   # platform activity + availability
    "location":     0.06,   # India preferred; Pune/Noida ideal
    "education":    0.04,   # soft signal
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# ─────────────────────────────────────────────────────────────────────────────
# Company & Industry Lists
# ─────────────────────────────────────────────────────────────────────────────

# JD explicitly disqualifies candidates whose ENTIRE career is at these firms.
# We penalise per-role if the company appears in this list.
IT_SERVICES_COMPANIES = frozenset({
    "tcs", "tata consultancy", "tata consultancy services",
    "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "techmahindra",
    "hcl", "hcltech", "hcl technologies",
    "mphasis", "hexaware", "mindtree", "ltimindtree", "lti mindtree",
    "l&t infotech", "l&t technology",
    "niit technologies", "zensar", "mastech",
    "persistent systems", "persistent",
    "cyient", "sonata software", "kpit",
    "happiest minds", "birlasoft", "firstsource",
    "coforge", "sasken", "ntt data", "dxc", "atos",
    "cgi group", "unisys", "syntel", "igate", "patni",
    "mphasis", "hexaware", "merce", "infotech",
})

# Strong positive: product companies relevant to the JD domain
PRODUCT_COMPANIES = frozenset({
    "google", "meta", "facebook", "amazon", "microsoft", "apple", "netflix",
    "uber", "airbnb", "stripe", "databricks", "snowflake", "confluent",
    "flipkart", "zomato", "swiggy", "ola", "razorpay", "paytm", "cred",
    "meesho", "zepto", "blinkit", "phonepe", "freshworks", "zoho",
    "browserstack", "chargebee", "postman", "druva", "icertis",
    "sarvam", "krutrim", "observe.ai", "uniphore", "haptik",
    "mad street den", "sigmoid", "fractal analytics", "fractal",
    "tiger analytics", "mu sigma", "sprinklr", "darwinbox", "skillate",
    "redrob", "khatabook", "delhivery", "nykaa",
    "groww", "upstox", "zerodha", "angel one", "smallcase",
    "cleartax", "urban company", "licious", "curefit",
    "pied piper", "initech", "hooli",                   # fictional but non-IT-services
    "acme corp", "wayne enterprises", "dunder mifflin", # fictional — neutral not positive
    "stark industries",                                  # fictional — neutral
})

# Positive-signal industries
GOOD_INDUSTRIES = frozenset({
    "ai/ml", "ml", "artificial intelligence",
    "fintech", "financial technology",
    "saas", "software", "software as a service",
    "e-commerce", "ecommerce",
    "edtech", "healthtech", "health tech",
    "developer tools", "devtools",
    "food delivery", "transportation", "logistics",
    "media tech", "adtech", "marketplace",
    "product", "deep tech",
})

# Negative-signal industries
BAD_INDUSTRIES = frozenset({
    "it services", "information technology services",
    "consulting", "management consulting",
    "outsourcing", "bpo", "ites",
    "business process management", "staffing",
})

# ─────────────────────────────────────────────────────────────────────────────
# Skills taxonomy
# ─────────────────────────────────────────────────────────────────────────────

# Tier 1: directly required by JD
TIER1_SKILLS = frozenset({
    # Vector/embedding retrieval
    "sentence transformers", "sentence-transformers",
    "embeddings", "vector search", "vector database", "vector databases",
    "semantic search", "dense retrieval",
    "faiss", "pinecone", "qdrant", "weaviate", "milvus", "chroma", "chromadb",
    "pgvector", "vespa", "annoy",
    "elasticsearch", "opensearch", "solr",
    "bm25", "hybrid search", "information retrieval", "sparse retrieval",
    # Evaluation / ranking
    "ndcg", "mrr", "map", "mean average precision",
    "ranking evaluation", "learning to rank", "ltr",
    "reranking", "re-ranking", "cross-encoder", "bi-encoder",
    # Embedding models
    "bge", "e5 embeddings", "ada embeddings",
})

# Tier 2: important and JD-relevant
TIER2_SKILLS = frozenset({
    "hugging face", "hugging face transformers", "transformers", "hf transformers",
    "pytorch", "nlp", "natural language processing",
    "llm", "large language models", "rag", "retrieval augmented generation",
    "fine-tuning", "fine-tuning llms", "lora", "qlora", "peft",
    "xgboost", "lightgbm", "gradient boosting",
    "recommendation systems", "collaborative filtering", "matrix factorization",
    "mlflow", "wandb", "weights & biases",
    "mlops", "model serving", "model deployment", "bentoml",
    "a/b testing", "ab testing", "experiment design",
    "feature engineering", "feature store",
    "python", "scikit-learn", "sklearn",
    "bert", "gpt", "t5", "llama", "mistral",
    "langchain", "llamaindex", "llama index",
    "prompt engineering",
})

# Tier 3: nice-to-have
TIER3_SKILLS = frozenset({
    "tensorflow", "keras", "spacy",
    "redis", "kafka", "spark", "pyspark", "dbt", "airflow",
    "docker", "kubernetes", "k8s",
    "aws", "gcp", "google cloud", "azure", "cloud",
    "sql", "postgresql", "mysql",
    "golang", "go", "rust", "scala",
    "kubeflow", "mlflow", "ray",
})

# Anti-skills: if these dominate a profile it's a bad signal for this role
ANTI_SKILLS = frozenset({
    "photoshop", "figma", "illustrator", "indesign", "canva",
    "marketing", "sales", "content writing", "seo", "sem",
    "project management", "scrum", "agile", "jira", "confluence",
    "sap", "crm", "salesforce", "erp", "oracle",
    "autocad", "solidworks", "catia",
    "accounting", "finance", "tally",
    "javascript", "react", "angular", "vue", "node.js", "tailwind",
    "typescript", "graphql", "grpc", "spring boot", "java",
})

# Skills that indicate the JD-critical domain even without exact match
DOMAIN_KEYWORDS_IN_DESCRIPTION = [
    "embeddings", "vector", "retrieval", "ranking", "recommendation",
    "search", "nlp", "language model", "llm", "fine-tun",
    "semantic", "similarity", "indexing", "rerank", "bi-encoder",
    "cross-encoder", "dense", "sparse", "bm25", "faiss",
    "pinecone", "qdrant", "weaviate", "milvus", "elasticsearch",
    "ndcg", "mrr", "precision@", "recall@", "a/b test",
    "learning to rank", "ltr", "feature engineering",
    "production ml", "deployed", "shipped",
]

# ─────────────────────────────────────────────────────────────────────────────
# Proficiency → numeric mapping
# ─────────────────────────────────────────────────────────────────────────────
PROFICIENCY_VALUE = {
    "beginner":     0.30,
    "intermediate": 0.60,
    "advanced":     0.85,
    "expert":       1.00,
}

# ─────────────────────────────────────────────────────────────────────────────
# ML title keywords (positive for this JD)
# ─────────────────────────────────────────────────────────────────────────────
ML_TITLE_TERMS = frozenset({
    "ml engineer", "machine learning engineer", "machine learning",
    "ai engineer", "artificial intelligence",
    "nlp engineer", "nlp",
    "data scientist", "applied scientist", "research engineer",
    "recommendation", "recommendation systems",
    "search engineer", "ranking engineer", "retrieval engineer",
    "applied ml", "applied ai",
    "llm engineer", "generative ai",
    "deep learning", "computer vision engineer",   # CV is OK if NLP exposure exists
})

# Non-IC / non-technical titles (negative)
NON_IC_TITLE_TERMS = frozenset({
    "operations manager", "operations",
    "hr manager", "human resources",
    "marketing manager", "marketing",
    "sales manager", "sales",
    "business analyst", "business development",
    "product manager",   # PM ≠ engineer
    "account manager", "accountant",
    "project manager",
    "customer support", "customer success",
    "graphic designer", "ux designer", "designer",
    "content writer", "content strategist",
    "civil engineer", "mechanical engineer",
})

# ─────────────────────────────────────────────────────────────────────────────
# Location preferences
# ─────────────────────────────────────────────────────────────────────────────
PREFERRED_CITIES = frozenset({
    "noida", "delhi ncr", "delhi", "new delhi",
    "gurgaon", "gurugram", "faridabad",
    "pune", "pimpri", "hinjawadi",
    "hyderabad", "secunderabad",
    "bangalore", "bengaluru",
    "mumbai", "navi mumbai", "thane",
    "chennai", "kolkata",
})

# ─────────────────────────────────────────────────────────────────────────────
# Education
# ─────────────────────────────────────────────────────────────────────────────
RELEVANT_EDU_FIELDS = frozenset({
    "computer science", "computer engineering",
    "information technology", "information science",
    "artificial intelligence", "machine learning", "data science",
    "electrical engineering", "electronics", "electronics and communication",
    "mathematics", "statistics", "computational mathematics",
    "software engineering",
})

EDU_TIER_SCORE = {
    "tier_1": 100,
    "tier_2": 75,
    "tier_3": 55,
    "tier_4": 40,
    "unknown": 45,
}

# ─────────────────────────────────────────────────────────────────────────────
# Behavioral / availability thresholds
# ─────────────────────────────────────────────────────────────────────────────
# Days since last active → score delta
ACTIVITY_SCORE = [
    (14,  +20),
    (30,  +15),
    (60,  +8),
    (90,  +0),
    (180, -15),
    (365, -25),
    (999, -35),
]

# ─────────────────────────────────────────────────────────────────────────────
# Honeypot detection thresholds
# ─────────────────────────────────────────────────────────────────────────────
HONEYPOT_HIGH_ENDORSE_ZERO_DUR_THRESHOLD = 15   # endorsements > X with duration == 0
HONEYPOT_EXPERT_LOW_ASSESS_THRESHOLD     = 25   # expert + assessment < X
HONEYPOT_ADVANCED_LOW_ASSESS_THRESHOLD   = 15   # advanced + assessment < X
HONEYPOT_MASS_EXPERT_COUNT               = 8    # more than N expert skills total

# ─────────────────────────────────────────────────────────────────────────────
# Salary budget (INR LPA) — used as soft filter
# Redrob is a Series A company; likely budget ~30-60 LPA for this role
# ─────────────────────────────────────────────────────────────────────────────
SALARY_BUDGET_MAX_LPA = 80.0   # Candidates expecting > this are penalised
SALARY_BUDGET_MIN_LPA = 15.0   # Candidates expecting < this signal mismatch

# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF re-ranking settings
# ─────────────────────────────────────────────────────────────────────────────
TFIDF_TOP_N_FOR_RERANK = 500    # Score all candidates, re-rank top-N with TF-IDF
TFIDF_BLEND_WEIGHT     = 0.25   # structured=0.75, tfidf=0.25  (increased for better top-10 separation)
TFIDF_MAX_FEATURES     = 8000
TFIDF_NGRAM_RANGE      = (1, 2)