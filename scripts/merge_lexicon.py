"""Merge Gemini-generated terms with existing lexicon.

Analyzes the generated terms and outputs additions by category.

Usage:
    cd /Users/omega/Projects/ProfessorGemini
    source .venv/bin/activate
    python scripts/merge_lexicon.py
"""

import json
from pathlib import Path
from collections import defaultdict


def normalize(term: str) -> str:
    """Normalize term for comparison."""
    return term.lower().strip().replace("-", " ").replace("_", " ").replace("(", "").replace(")", "")


# Existing terms from the lexicon page (extracted)
EXISTING_TERMS = {
    # Reliability & SLA Engineering
    normalize("Composite SLA"),
    normalize("Error Budget"),
    normalize("Burn Rate"),
    normalize("SLI/SLO/SLA Pipeline"),
    normalize("Critical User Journey"),
    normalize("CUJ"),
    normalize("Gray Failure"),
    normalize("MTTR vs MTTF"),
    normalize("Multi-Window Alerting"),

    # Architectural Strategy
    normalize("Blast Radius"),
    normalize("Cell-Based Architecture"),
    normalize("Strangler Fig Pattern"),
    normalize("Expand-Contract Pattern"),
    normalize("Bulkhead Pattern"),
    normalize("Architectural Runway"),
    normalize("N+1 Redundancy"),

    # Data & Consistency
    normalize("Consistent Hashing"),
    normalize("Hot Partition"),
    normalize("Celebrity Problem"),
    normalize("Scatter-Gather"),
    normalize("Vector Clock"),
    normalize("Quorum"),
    normalize("SAGA Pattern"),

    # Program Execution & Risk
    normalize("Fix-Forward vs Rollback"),
    normalize("Dark Deployment"),
    normalize("Progressive Delivery"),
    normalize("Bake Time"),
    normalize("Feature Flag Taxonomy"),
    normalize("Flag Debt"),
    normalize("Go/No-Go Criteria"),

    # Cloud Economics & FinOps
    normalize("Unit Economics"),
    normalize("Commitment Coverage"),
    normalize("Stranded Capacity"),
    normalize("Cost Iceberg"),
    normalize("Network Egress Tax"),
    normalize("Chargeback vs Showback"),
    normalize("TCO"),
    normalize("Total Cost of Ownership"),

    # Incident Management
    normalize("Incident Commander"),
    normalize("IC"),
    normalize("Severity Matrix"),
    normalize("Blameless Postmortem"),
    normalize("5 Whys"),
    normalize("Swiss Cheese"),
    normalize("COE"),
    normalize("Correction of Error"),
    normalize("Action Item Governance"),

    # Organizational Design
    normalize("Conway's Law"),
    normalize("Inverse Conway Maneuver"),
    normalize("Team Topologies"),
    normalize("Cognitive Load"),
    normalize("Thinnest Viable Platform"),
    normalize("TVP"),
    normalize("KTLO Ratio"),

    # Technical Debt & Governance
    normalize("Fowler's Debt Quadrant"),
    normalize("Velocity Tax"),
    normalize("Toil"),
    normalize("20% Tax Rule"),
    normalize("Pain Index"),
}


# Category mappings for Gemini terms
CATEGORY_KEYWORDS = {
    "Reliability & SLA Engineering": ["sla", "slo", "sli", "reliability", "availability", "uptime", "error budget", "burn rate", "latency", "alert", "monitoring", "observability"],
    "Architectural Strategy": ["architecture", "pattern", "strangler", "bulkhead", "cell", "microservice", "monolith", "migration", "blast radius", "failover", "redundancy"],
    "Data & Consistency": ["data", "consistency", "replication", "sharding", "partition", "cache", "database", "consensus", "cap theorem", "eventual", "transaction", "cdc"],
    "Program Execution & Risk": ["deploy", "release", "rollback", "canary", "feature flag", "migration", "risk", "go/no-go", "progressive"],
    "Cloud Economics & FinOps": ["cost", "finops", "cloud", "reserved", "spot", "savings", "chargeback", "showback", "unit economics", "tco"],
    "Incident Management": ["incident", "postmortem", "rca", "root cause", "severity", "escalation", "commander", "coe", "pir"],
    "Organizational Design": ["team", "conway", "topology", "cognitive load", "platform", "ktlo", "organization"],
    "Technical Debt & Governance": ["debt", "toil", "governance", "compliance", "rfc", "adr"],
    "Security & Compliance": ["security", "zero trust", "mtls", "oauth", "gdpr", "pci", "soc", "compliance", "encryption", "authentication"],
    "ML/AI Systems": ["ml", "ai", "model", "inference", "training", "vector", "embedding", "llm", "rag", "feature store"],
    "API Design": ["api", "idl", "protobuf", "grpc", "graphql", "rest", "versioning", "deprecation"],
    "Performance & Scaling": ["scaling", "auto-scaling", "load balancing", "rate limiting", "backpressure", "throughput", "qps"],
}


def categorize_term(term: str, definition: str) -> str:
    """Attempt to categorize a term based on keywords."""
    combined = f"{term} {definition}".lower()

    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in combined)
        if score > 0:
            scores[category] = score

    if scores:
        return max(scores, key=scores.get)
    return "Uncategorized"


def main():
    # Load Gemini terms
    json_path = Path(__file__).parent.parent / "lexicon_terms.json"
    if not json_path.exists():
        print("ERROR: lexicon_terms.json not found. Run batch_lexicon.py first.")
        return

    data = json.loads(json_path.read_text())

    # Find unique terms not in existing lexicon
    unique_terms = []
    for doc in data["documents"]:
        for term in doc.get("terms", []):
            normalized = normalize(term["term"])
            if normalized not in EXISTING_TERMS and len(normalized) > 2:
                unique_terms.append({
                    "term": term["term"],
                    "definition": term["definition"],
                    "source": doc["filename"],
                    "category": categorize_term(term["term"], term["definition"]),
                })

    # Deduplicate by normalized term
    seen = set()
    deduped = []
    for t in unique_terms:
        norm = normalize(t["term"])
        if norm not in seen:
            seen.add(norm)
            deduped.append(t)

    # Group by category
    by_category = defaultdict(list)
    for t in deduped:
        by_category[t["category"]].append(t)

    # Print results
    print("=" * 70)
    print("GEMINI LEXICON ADDITIONS (Not in Existing Lexicon)")
    print("=" * 70)
    print(f"\nTotal unique additions: {len(deduped)}")
    print(f"Existing terms: {len(EXISTING_TERMS)}")
    print()

    # Sort categories by count
    sorted_cats = sorted(by_category.items(), key=lambda x: len(x[1]), reverse=True)

    for category, terms in sorted_cats:
        print(f"\n{'=' * 70}")
        print(f"## {category} ({len(terms)} terms)")
        print("-" * 70)
        for t in terms[:10]:  # Top 10 per category
            print(f"\n**{t['term']} ***")
            print(f"  {t['definition']}")
            print(f"  Source: {t['source']}")

    # Output JSON for programmatic use
    output_path = Path(__file__).parent.parent / "lexicon_additions.json"
    output = {
        "total_additions": len(deduped),
        "by_category": {cat: terms for cat, terms in sorted_cats},
    }
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\n\nFull output saved to: {output_path}")


if __name__ == "__main__":
    main()
