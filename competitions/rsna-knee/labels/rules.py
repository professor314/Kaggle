"""Clinical report-label rules (Tier 0.1) — text -> 12 finding probabilities.

Improves the keyword baseline with the clinical logic from research_notes.md:
- Unmentioned finding = 0 (radiologists report positives, stay silent on negatives).
- Explicit negation ("no effusion", "sin rotura") = 0.
- Impression/Conclusion section weighted highest.
- Current findings only (chronic/post-op handled where possible).
- Synovitis breadth (Hoffa impingement, plica, friction syndromes).
- Contusion needs a traumatic pattern (not degenerative marrow edema).
- Fracture includes avulsion + insufficiency fractures.

Pure functions, no Kaggle/torch dependency, so they unit-test locally and import
into the offline label kernel. Returns a probability in [0,1] per finding.
"""
from __future__ import annotations

import re
from typing import Dict, List

FINDINGS: List[str] = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

# Structure findings: positive only when injury language appears near the mention.
STRUCT_CUES: Dict[str, List[str]] = {
    "ACL": ["acl", "anterior cruciate", "lca", "cruzado anterior", "ligamento cruzado anterior"],
    "MCL": ["mcl", "medial collateral", "lcm", "colateral medial", "ligamento lateral interno"],
    "Medial Meniscus": ["medial meniscus", "menisco medial", "menisco interno"],
    "Lateral Meniscus": ["lateral meniscus", "menisco lateral", "menisco externo"],
}
# Presence findings: positive if mentioned and not negated (compartment-scoped for OA).
PRESENCE_CUES: Dict[str, List[str]] = {
    "Medial OA": ["medial compartment osteoarth", "medial osteoarth",
                  "artrosis femorotibial medial", "medial femorotibial",
                  "medial compartment chondral", "medial compartment cartilage",
                  "chondrosis medial", "medial chondral", "condral medial",
                  "gonartrosis medial"],
    "Lateral OA": ["lateral compartment osteoarth", "lateral osteoarth",
                   "artrosis femorotibial lateral", "lateral femorotibial",
                   "lateral compartment chondral", "lateral compartment cartilage",
                   "chondrosis lateral", "lateral chondral", "condral lateral",
                   "gonartrosis lateral"],
    "PF OA": ["patellofemoral osteoarth", "patellofemoral chondr", "patellar facet",
              "patellofemoral", "femoropatelar", "patelofemoral", "chondromalacia",
              "condromalacia", "trochlear chondr", "retropatellar chondr"],
    "Effusion": ["effusion", "joint fluid", "fluid accumulation", "derrame",
                 "fluid distension", "hydrarthrosis", "hidrartrosis", "joint distension"],
    "Synovitis": ["synovitis", "synovial thickening", "synovial membrane thick",
                  "sinovitis", "sinovial", "synovial proliferation",
                  "hoffa impingement", "hoffa's fat pad edema", "plica",
                  "friction syndrome", "fat pad impingement"],
    "Baker's": ["baker", "popliteal cyst", "quiste de baker", "quiste popliteo",
                "popliteal fossa cyst"],
    "Contusion": ["contusion", "bone bruise", "bone contusion", "traumatic marrow edema",
                  "contusion osea", "contusion medular", "trabecular microfracture"],
    "Fracture": ["fracture", "fractura", "avulsion fracture", "insufficiency fracture",
                 "avulsion osea", "fractura por insuficiencia", "stress fracture",
                 "fractura de estres"],
}
INJURY = ["tear", "torn", "rupture", "ruptur", "rotura", "roto", "rota", "desgarro",
          "lesion", "sprain", "esguince", "injury", "disrupt", "chondral defect",
          "cartilage defect", "chondrosis", "chondromalacia", "signal", "senal"]
NEG = ["no ", "not ", "intact", "preserved", "normal", "without", "unremarkable",
       "sin ", "conservad", "not torn", "negative", "descarta", "no evidence",
       "is not", "are not", "no significant", "no acute", "sin signos", "sin evidencia",
       "integr", "no rotura", "no se observa", "no hay"]
# Degenerative context that should DOWN-weight a "marrow edema" contusion call.
DEGEN = ["degenerative", "osteoarth", "chronic", "artrosis", "subchondral cyst"]


def _impression(text: str) -> str:
    """Return the Impression/Conclusion tail if present, else the whole text."""
    t = str(text)
    for kw in ["impression", "impresion", "conclusion", "conclusión", "impresión"]:
        m = re.search(kw, t, flags=re.IGNORECASE)
        if m:
            return t[m.start():]
    return t


def _mentioned(t: str, cues: List[str]) -> int:
    for cue in cues:
        i = t.find(cue)
        if i != -1:
            return i
    return -1


def score_finding(text: str, finding: str) -> float:
    """Probability in [0,1] that `finding` is positive in `text`.

    Unmentioned -> 0.0 (radiologists omit negatives). Impression section, if
    present, doubles a positive's confidence toward 1.0.
    """
    raw = " " + str(text).lower().replace("\n", " ") + " "
    imp = " " + _impression(str(text)).lower().replace("\n", " ") + " "

    if finding in PRESENCE_CUES:
        cues = PRESENCE_CUES[finding]
        best = 0.0
        found = False
        for cue in cues:
            idx = raw.find(cue)
            while idx != -1:
                found = True
                window = raw[max(0, idx - 45): idx + 55]
                neg = any(n in window for n in NEG)
                score = 0.05 if neg else 0.85
                if finding == "Contusion" and not neg:
                    # require a traumatic pattern; degenerative edema -> down-weight
                    if any(d in window for d in DEGEN):
                        score = 0.3
                if not neg and any(cue in imp for cue in cues):
                    score = 0.95
                best = max(best, score)
                idx = raw.find(cue, idx + 1)
        return best if found else 0.0

    # structure findings
    cues = STRUCT_CUES[finding]
    hit = _mentioned(raw, cues)
    if hit == -1:
        return 0.0
    window = raw[max(0, hit - 70): hit + 90]
    neg = any(n in window for n in NEG)
    inj = any(w in window for w in INJURY)
    # A "normal/intact/preserved" statement about the structure is a strong
    # negative even if an injury word (e.g. "tear") appears — it's being denied.
    strong_neg = any(s in window for s in
                     ("intact", "preserved", "normal", "unremarkable", "integr",
                      "conservad", "no tear", "not torn", "no rotura"))
    if inj and not neg:
        return 0.95 if _mentioned(imp, cues) != -1 else 0.9
    if neg and not inj:
        return 0.05
    if inj and neg:
        return 0.05 if strong_neg else 0.5
    return 0.15  # mentioned, no clear injury language


def score_report(text: str, findings: List[str] = FINDINGS) -> Dict[str, float]:
    """Return {finding: prob} for all findings in one report."""
    return {f: score_finding(text, f) for f in findings}
