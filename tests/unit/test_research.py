"""Unit tests for ResearchDocumentGenerator module.

Tests:
1. test_create_research_document_has_sections — output file contains all section headers
2. test_create_design_doc_includes_findings — output contains provided hypothesis text
3. test_files_created_in_correct_directory — files exist at expected paths

Uses tmp_path for all file operations.

Validates: Requirements 4.1, 4.2, 4.3
"""

import os

import pytest

from kaggle_ml_toolkit.research import ResearchDocumentGenerator


@pytest.fixture
def generator():
    return ResearchDocumentGenerator()


class TestCreateResearchDocumentHasSections:
    """Research document contains all required section headers."""

    def test_create_research_document_has_sections(self, generator, tmp_path):
        output_dir = str(tmp_path / "research")
        path = generator.create_research_document("Test Competition", output_dir)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        required_sections = [
            "## Domain Background",
            "## Prior Work",
            "## Hypotheses",
            "## Relevant Datasets",
            "## Recommended Approaches",
            "## Domain Knowledge",
        ]
        for section in required_sections:
            assert section in content, f"Missing section: {section}"

    def test_research_document_contains_competition_name(self, generator, tmp_path):
        output_dir = str(tmp_path / "research")
        path = generator.create_research_document("Titanic Survival", output_dir)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "Titanic Survival" in content


class TestCreateDesignDocIncludesFindings:
    """Design document includes provided hypothesis text."""

    def test_create_design_doc_includes_findings(self, generator, tmp_path):
        output_dir = str(tmp_path / "design")
        findings = {
            "primary_hypothesis": "Passenger class and age are the strongest survival predictors.",
            "modeling_strategy": "Ensemble of gradient boosting and logistic regression.",
            "evaluation_priorities": "Use stratified 5-fold CV with accuracy as primary metric.",
            "content_goals": "Publish a blog post explaining feature importance.",
            "decisions": ["Use XGBoost as primary model", "Target encode high-cardinality features"],
        }

        path = generator.create_design_doc("Test Competition", findings, output_dir)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "Passenger class and age are the strongest survival predictors." in content
        assert "Ensemble of gradient boosting and logistic regression." in content
        assert "Use XGBoost as primary model" in content
        assert "Target encode high-cardinality features" in content

    def test_design_doc_includes_evaluation_priorities(self, generator, tmp_path):
        output_dir = str(tmp_path / "design")
        findings = {
            "primary_hypothesis": "Feature X matters most.",
            "evaluation_priorities": "Optimize for F1 score.",
        }

        path = generator.create_design_doc("Competition", findings, output_dir)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "Optimize for F1 score." in content


class TestFilesCreatedInCorrectDirectory:
    """Files are created at expected paths."""

    def test_research_document_at_expected_path(self, generator, tmp_path):
        output_dir = str(tmp_path / "output")
        path = generator.create_research_document("Test", output_dir)

        expected_path = os.path.join(output_dir, "research_document.md")
        assert path == expected_path
        assert os.path.isfile(expected_path)

    def test_design_doc_at_expected_path(self, generator, tmp_path):
        output_dir = str(tmp_path / "output")
        findings = {"primary_hypothesis": "Test hypothesis."}
        path = generator.create_design_doc("Test", findings, output_dir)

        expected_path = os.path.join(output_dir, "design_doc.md")
        assert path == expected_path
        assert os.path.isfile(expected_path)

    def test_creates_output_directory_if_not_exists(self, generator, tmp_path):
        output_dir = str(tmp_path / "nested" / "deep" / "dir")
        generator.create_research_document("Test", output_dir)

        assert os.path.isdir(output_dir)
