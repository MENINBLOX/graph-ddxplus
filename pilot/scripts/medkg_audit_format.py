#!/usr/bin/env python3
"""Audit JSON output format for traceable diagnosis.

This module defines how a diagnosis output looks in our medkg system,
exposing source provenance + calibrated confidence per diagnosis.

Example output schema:
{
  "patient_id": "P_001",
  "evidences": ["chest pain", "shortness of breath", "fever"],
  "ranked_diagnoses": [
    {
      "rank": 1,
      "disease": "Pneumonia",
      "umls_cui": "C0032285",
      "raw_score": 0.87,
      "calibrated_confidence": 0.72,
      "supporting_features": [
        {
          "phenotype": "fever",
          "matched_evidence": "fever",
          "kg_score": 0.95,
          "n_sources": 4,
          "sources": ["statpearls", "medlineplus", "wikipedia", "orphanet"],
          "provenance": {
            "statpearls": [{"nbk": "NBK558903", "section": "history and physical"}],
            "medlineplus": [{"topic_url": "...", "topic_title": "Pneumonia"}],
            "wikipedia": [{"pageid": 52135, "revid": 12345, "section": "signs and symptoms"}],
            "orphanet": [{"orpha_code": "1234"}]
          }
        },
        ...
      ],
      "missing_features": [],
      "explanation": "..."
    },
    ...
  ],
  "audit_log": {
    "kg_version": "medkg_2026-05-06",
    "model": "gemma-4-E4B-it",
    "ie_prompt_version": "v3-principles-only",
    "calibration_method": "platt-scaling",
    "n_kg_edges": 119968,
    "decision_path": [
      "1. Loaded 49 DDXPlus disease candidates with KG features",
      "2. Stage 1 scoring: top-1 'Pneumonia' (raw=0.87)",
      "3. No tie-break needed (gap=0.18 vs top-2 'Bronchitis' raw=0.69)",
      "4. Calibrated to 0.72 confidence"
    ]
  }
}
"""
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional


@dataclass
class FeatureProvenance:
    phenotype: str
    matched_evidence: Optional[str]
    kg_score: float
    n_sources: int
    sources: List[str]
    hpo_id: Optional[str] = None
    provenance: Dict = field(default_factory=dict)


@dataclass
class DiagnosisRanked:
    rank: int
    disease: str
    umls_cui: Optional[str]
    raw_score: float
    calibrated_confidence: float
    supporting_features: List[FeatureProvenance] = field(default_factory=list)
    missing_features: List[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class AuditLog:
    kg_version: str
    model: str
    ie_prompt_version: str
    calibration_method: str
    n_kg_edges: int
    decision_path: List[str] = field(default_factory=list)


@dataclass
class DiagnosisAudit:
    patient_id: str
    evidences: List[str]
    ranked_diagnoses: List[DiagnosisRanked]
    audit_log: AuditLog


def example_audit():
    """Return an example audit JSON for demonstration."""
    audit = DiagnosisAudit(
        patient_id="P_001_demo",
        evidences=["chest pain", "shortness of breath", "fever", "cough"],
        ranked_diagnoses=[
            DiagnosisRanked(
                rank=1,
                disease="Pneumonia",
                umls_cui="C0032285",
                raw_score=0.87,
                calibrated_confidence=0.72,
                supporting_features=[
                    FeatureProvenance(
                        phenotype="fever",
                        matched_evidence="fever",
                        kg_score=0.95,
                        n_sources=4,
                        sources=["statpearls", "medlineplus", "wikipedia", "orphanet"],
                        provenance={
                            "statpearls": [{"nbk": "NBK558903", "section": "history and physical"}],
                            "medlineplus": [{"topic_title": "Pneumonia"}],
                            "wikipedia": [{"pageid": 52135, "section": "signs and symptoms"}],
                            "orphanet": [{"orpha_code": "1234"}]
                        }
                    ),
                    FeatureProvenance(
                        phenotype="cough",
                        matched_evidence="cough",
                        kg_score=0.87,
                        n_sources=3,
                        sources=["statpearls", "medlineplus", "wikipedia"],
                    ),
                ],
                missing_features=["pleuritic chest pain"],
                explanation="Patient evidences match 3/4 typical features (fever, cough, dyspnea); KG indicates 4-source agreement on these features."
            ),
        ],
        audit_log=AuditLog(
            kg_version="medkg_2026-05-06_multi-source",
            model="gemma-4-E4B-it",
            ie_prompt_version="v3-principles-only",
            calibration_method="platt-scaling",
            n_kg_edges=119968,
            decision_path=[
                "Loaded 49 DDXPlus disease candidates",
                "Stage 1 scoring with multi-source KG features",
                "Top-1 = Pneumonia (raw=0.87, calibrated=0.72)",
                "Gap to top-2 = 0.18, no tie-break",
                "Provenance: 4 sources for fever, 3 for cough",
            ]
        )
    )
    return audit


def to_json(audit: DiagnosisAudit) -> str:
    return json.dumps(asdict(audit), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print(to_json(example_audit()))
