"""Unit tests for ContentGenerator.

Validates: Requirements 16.1, 16.2, 16.3, 16.4, 16.5, 16.6, 22.1, 22.2, 22.3, 22.4, 22.5, 22.6, 25.1, 25.2, 25.3
"""

import os
import tempfile
from pathlib import Path

import pytest

from kaggle_ml_toolkit.content_generator import ContentGenerator


@pytest.fixture
def generator():
    """A ContentGenerator instance."""
    return ContentGenerator()


@pytest.fixture
def competition_dir(tmp_path):
    """Create a minimal competition directory structure with artifacts."""
    comp_dir = tmp_path / "test_competition"
    research_dir = comp_dir / "research"
    research_dir.mkdir(parents=True)
    content_dir = comp_dir / "content"
    content_dir.mkdir(parents=True)

    # Create minimal research artifacts
    (research_dir / "research_document.md").write_text(
        "# Research Document\n\n## Domain Background\nThis is about classification.",
        encoding="utf-8",
    )
    (research_dir / "design_doc.md").write_text(
        "# Design Document\n\n## Strategy\nUse gradient boosting with engineered features.",
        encoding="utf-8",
    )

    return str(comp_dir)


class TestGenerateBlogPostReturnsMarkdown:
    """test_generate_blog_post_returns_markdown — output starts with '#' and contains key sections."""

    def test_generate_blog_post_returns_markdown(self, generator, competition_dir):
        """Verify blog post output starts with '#' and contains key structural sections."""
        result = generator.generate_blog_post(competition_dir)

        # Starts with a markdown heading
        assert result.startswith("#"), "Blog post should start with a markdown heading"

        # Contains key sections
        assert "## Introduction" in result
        assert "## Domain Research Summary" in result
        assert "## Exploratory Data Analysis" in result
        assert "## Feature Engineering" in result
        assert "## Model Selection" in result
        assert "## Optimization" in result
        assert "## Final Performance" in result


class TestGenerateKaggleWriteupHasAllSections:
    """test_generate_kaggle_writeup_has_all_sections — Summary, Approach, What Worked, etc."""

    def test_generate_kaggle_writeup_has_all_sections(self, generator, competition_dir, tmp_path):
        """Verify the Kaggle writeup contains all required standard sections."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            writeup_path = generator.generate_kaggle_writeup(
                competition_dir=competition_dir,
                final_score=0.8234,
                baseline_score=0.6500,
            )

            content = Path(writeup_path).read_text(encoding="utf-8")

            required_sections = [
                "## Summary",
                "## Approach",
                "## What Worked",
                "## What Didn't Work",
                "## Final Model Description",
                "## Score",
                "## AI Disclosure",
            ]
            for section in required_sections:
                assert section in content, f"Missing section: {section}"
        finally:
            os.chdir(original_cwd)


class TestKaggleWriteupIncludesScores:
    """test_kaggle_writeup_includes_scores — final_score and baseline_score appear in text."""

    def test_kaggle_writeup_includes_scores(self, generator, competition_dir, tmp_path):
        """Verify both final_score and baseline_score numeric values appear in the writeup."""
        final_score = 0.7891
        baseline_score = 0.5432

        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            writeup_path = generator.generate_kaggle_writeup(
                competition_dir=competition_dir,
                final_score=final_score,
                baseline_score=baseline_score,
            )

            content = Path(writeup_path).read_text(encoding="utf-8")

            assert f"{final_score:.4f}" in content, (
                f"Final score {final_score:.4f} not found in writeup"
            )
            assert f"{baseline_score:.4f}" in content, (
                f"Baseline score {baseline_score:.4f} not found in writeup"
            )
        finally:
            os.chdir(original_cwd)


class TestGenerateMetaArticleCreatesFile:
    """test_generate_meta_article_creates_file — file exists at expected path."""

    def test_generate_meta_article_creates_file(self, generator, tmp_path):
        """Verify generate_meta_article creates a file at the expected location."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result_path = generator.generate_meta_article(
                topic="toolkit-intro",
                output_dir=str(tmp_path / "docs" / "blog"),
            )

            assert Path(result_path).exists(), (
                f"Meta article file not found at {result_path}"
            )
            # Filename should be derived from the topic
            assert "toolkit-intro.md" in result_path
        finally:
            os.chdir(original_cwd)


class TestDualSaveIdenticalContent:
    """test_dual_save_identical_content — both files have same content."""

    def test_dual_save_identical_content(self, generator, tmp_path):
        """Verify save_with_dual_write produces byte-identical files in both locations."""
        content = "# My Article\n\nSome great content here.\n"
        comp_content_path = str(tmp_path / "comp" / "content")

        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            comp_path, blog_path = generator.save_with_dual_write(
                content=content,
                competition_content_path=comp_content_path,
                filename="test_article.md",
                title="Test Article",
                description="A test article for dual-save",
            )

            # Both files should exist
            assert Path(comp_path).exists()
            assert Path(blog_path).exists()

            # Content should be byte-identical
            comp_bytes = Path(comp_path).read_bytes()
            blog_bytes = Path(blog_path).read_bytes()
            assert comp_bytes == blog_bytes, (
                "Competition and blog files should be byte-identical"
            )
        finally:
            os.chdir(original_cwd)


class TestUpdateBlogIndexAddsEntry:
    """test_update_blog_index_adds_entry — entry appears in docs/blog/README.md."""

    def test_update_blog_index_adds_entry(self, generator, tmp_path):
        """Verify update_blog_index creates/updates docs/blog/README.md with new entry."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            generator.update_blog_index(
                title="My New Article",
                date="2024-06-15",
                description="An article about ML",
                filename="my-new-article.md",
            )

            index_path = Path("docs/blog/README.md")
            assert index_path.exists(), "Blog index README.md was not created"

            content = index_path.read_text(encoding="utf-8")
            assert "My New Article" in content
            assert "2024-06-15" in content
            assert "my-new-article.md" in content
            assert "An article about ML" in content
        finally:
            os.chdir(original_cwd)

    def test_update_blog_index_appends_to_existing(self, generator, tmp_path):
        """Verify a second call appends rather than overwrites."""
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            generator.update_blog_index(
                title="First Article",
                date="2024-01-01",
                description="First",
                filename="first.md",
            )
            generator.update_blog_index(
                title="Second Article",
                date="2024-02-01",
                description="Second",
                filename="second.md",
            )

            index_path = Path("docs/blog/README.md")
            content = index_path.read_text(encoding="utf-8")

            assert "First Article" in content
            assert "Second Article" in content
        finally:
            os.chdir(original_cwd)


class TestAudienceLevelsChangeContent:
    """test_audience_levels_change_content — beginner vs advanced produce different text."""

    def test_audience_levels_change_content(self, generator, competition_dir):
        """Verify beginner and advanced audience levels produce different blog post content."""
        beginner_post = generator.generate_blog_post(
            competition_dir=competition_dir,
            target_audience="beginner",
        )
        advanced_post = generator.generate_blog_post(
            competition_dir=competition_dir,
            target_audience="advanced",
        )

        # The posts should be different (audience impacts content depth)
        assert beginner_post != advanced_post, (
            "Beginner and advanced posts should produce different content"
        )

        # Beginner content should use more explanatory language
        # (checking the introduction section which varies by audience)
        assert "explain" in beginner_post.lower() or "detail" in beginner_post.lower(), (
            "Beginner post should contain explanatory language"
        )
        assert "technical" in advanced_post.lower(), (
            "Advanced post should contain technical language"
        )
