"""Research document generation for Kaggle ML Toolkit.

Creates structured research and design documents for competition projects,
guiding the research-first workflow with appropriate sections and placeholders.
"""

import os
from datetime import date
from typing import Any, Dict


class ResearchDocumentGenerator:
    """Creates structured research documents for competitions.

    Generates markdown documents with predefined sections to guide
    domain research and design decisions before modeling begins.
    """

    def create_research_document(
        self, competition_name: str, output_dir: str
    ) -> str:
        """Create initial research document with required sections.

        Generates a markdown template with sections for: Domain Background,
        Prior Work, Hypotheses, Relevant Datasets, Recommended Approaches,
        and Domain Knowledge.

        Args:
            competition_name: Name of the Kaggle competition.
            output_dir: Directory where the document will be written.
                Created if it does not exist.

        Returns:
            Path to the created research document.
        """
        os.makedirs(output_dir, exist_ok=True)

        file_path = os.path.join(output_dir, "research_document.md")
        today = date.today().isoformat()

        content = f"""# Research Document: {competition_name}

**Created:** {today}

---

## Domain Background

_Describe the problem domain. What real-world context does this competition address? What are the key concepts a practitioner should understand?_

---

## Prior Work

_Summarize relevant prior solutions, papers, or Kaggle discussions. What approaches have been tried before? What scores did top solutions achieve?_

---

## Hypotheses

_List testable hypotheses about what will drive performance in this competition. Each hypothesis should be specific and falsifiable._

1. 
2. 
3. 

---

## Relevant Datasets

_Identify external datasets or supplementary data sources that could improve model performance. Note any licensing or usage restrictions._

---

## Recommended Approaches

_Based on domain research, list modeling approaches ranked by expected effectiveness. Include rationale for each recommendation._

1. 
2. 
3. 

---

## Domain Knowledge

_Capture domain-specific insights that inform feature engineering, data cleaning, or evaluation strategy. What do experts in this field know that could translate into better predictions?_

"""

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return file_path

    def create_design_doc(
        self,
        competition_name: str,
        research_findings: Dict[str, Any],
        output_dir: str,
    ) -> str:
        """Create design document from research findings.

        Generates a structured design document with sections for: Primary
        Hypothesis, Modeling Strategy, Evaluation Priorities, Content Goals,
        and Decisions Log.

        Args:
            competition_name: Name of the Kaggle competition.
            research_findings: Dictionary containing research outputs. Supports
                keys: 'primary_hypothesis', 'modeling_strategy',
                'evaluation_priorities', 'content_goals', 'decisions'.
            output_dir: Directory where the document will be written.
                Created if it does not exist.

        Returns:
            Path to the created design document.
        """
        os.makedirs(output_dir, exist_ok=True)

        file_path = os.path.join(output_dir, "design_doc.md")
        today = date.today().isoformat()

        primary_hypothesis = research_findings.get(
            "primary_hypothesis",
            "_To be determined from research._",
        )
        modeling_strategy = research_findings.get(
            "modeling_strategy",
            "_To be determined based on research findings._",
        )
        evaluation_priorities = research_findings.get(
            "evaluation_priorities",
            "_Define evaluation metrics and validation strategy._",
        )
        content_goals = research_findings.get(
            "content_goals",
            "_Define blog post topics and educational outputs._",
        )
        decisions = research_findings.get("decisions", [])

        decisions_section = ""
        if decisions:
            for decision in decisions:
                decisions_section += f"- {decision}\n"
        else:
            decisions_section = "_No decisions recorded yet._\n"

        content = f"""# Design Document: {competition_name}

**Created:** {today}

---

## Primary Hypothesis

{primary_hypothesis}

---

## Modeling Strategy

{modeling_strategy}

---

## Evaluation Priorities

{evaluation_priorities}

---

## Content Goals

{content_goals}

---

## Decisions Log

{decisions_section}
"""

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return file_path
