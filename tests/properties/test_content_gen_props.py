"""Property-based tests for ContentGenerator.

Validates: Requirements 22.1, 22.3, 22.6, 25.2, 25.3
"""

import os
import tempfile
from pathlib import Path

from hypothesis import given, settings, strategies as st

from kaggle_ml_toolkit.content_generator import ContentGenerator


# ---------------------------------------------------------------------------
# Property 34: Kaggle Writeup Format Completeness
# Feature: kaggle-ml-toolkit, Property 34
# ---------------------------------------------------------------------------

REQUIRED_WRITEUP_SECTIONS = [
    "## Summary",
    "## Approach",
    "## What Worked",
    "## What Didn't Work",
    "## Final Model Description",
    "## Score",
    "## AI Disclosure",
]


@given(
    final_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    baseline_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=20)
def test_kaggle_writeup_format_completeness(final_score, baseline_score):
    """For any valid competition dir with artifacts, the generated writeup contains
    all required sections (Summary, Approach, What Worked, What Didn't Work,
    Final Model Description, Score) AND an AI Disclosure section.

    **Validates: Requirements 22.1, 22.6**
    """
    generator = ContentGenerator()

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a minimal competition directory with artifacts
        comp_dir = Path(tmp_dir) / "test_comp"
        research_dir = comp_dir / "research"
        research_dir.mkdir(parents=True)
        content_dir = comp_dir / "content"
        content_dir.mkdir(parents=True)

        # Create minimal research artifacts
        (research_dir / "research_document.md").write_text(
            "# Research\nSome findings.", encoding="utf-8"
        )
        (research_dir / "design_doc.md").write_text(
            "# Design\nSome design decisions.", encoding="utf-8"
        )

        # Create docs/blog in temp dir (monkeypatch cwd)
        original_cwd = os.getcwd()
        os.chdir(tmp_dir)
        try:
            writeup_path = generator.generate_kaggle_writeup(
                competition_dir=str(comp_dir),
                final_score=final_score,
                baseline_score=baseline_score,
            )

            # Read the generated writeup content
            writeup_content = Path(writeup_path).read_text(encoding="utf-8")

            # Assert all required sections are present
            for section in REQUIRED_WRITEUP_SECTIONS:
                assert section in writeup_content, (
                    f"Missing required section '{section}' in writeup"
                )
        finally:
            os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Property 35: Writeup Score Inclusion
# Feature: kaggle-ml-toolkit, Property 35
# ---------------------------------------------------------------------------


@given(
    final_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    baseline_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=20)
def test_writeup_score_inclusion(final_score, baseline_score):
    """For any final_score and baseline_score, both numeric values appear
    in the generated markdown.

    **Validates: Requirements 22.3**
    """
    generator = ContentGenerator()

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a minimal competition directory
        comp_dir = Path(tmp_dir) / "score_comp"
        research_dir = comp_dir / "research"
        research_dir.mkdir(parents=True)
        content_dir = comp_dir / "content"
        content_dir.mkdir(parents=True)

        (research_dir / "research_document.md").write_text("# Research", encoding="utf-8")

        original_cwd = os.getcwd()
        os.chdir(tmp_dir)
        try:
            writeup_path = generator.generate_kaggle_writeup(
                competition_dir=str(comp_dir),
                final_score=final_score,
                baseline_score=baseline_score,
            )

            writeup_content = Path(writeup_path).read_text(encoding="utf-8")

            # Both scores should appear formatted to 4 decimal places
            assert f"{final_score:.4f}" in writeup_content, (
                f"Final score {final_score:.4f} not found in writeup"
            )
            assert f"{baseline_score:.4f}" in writeup_content, (
                f"Baseline score {baseline_score:.4f} not found in writeup"
            )
        finally:
            os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Property 37: Dual-Save Content Identity
# Feature: kaggle-ml-toolkit, Property 37
# ---------------------------------------------------------------------------


@given(
    content=st.text(min_size=10, max_size=500, alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
    )),
    filename=st.text(
        min_size=3, max_size=20,
        alphabet=st.characters(whitelist_categories=("Ll", "Nd")),
    ).map(lambda x: x + ".md"),
)
@settings(max_examples=20)
def test_dual_save_content_identity(content, filename):
    """The file saved to competition dir and docs/blog/ should be byte-identical.

    **Validates: Requirements 25.2**
    """
    generator = ContentGenerator()

    with tempfile.TemporaryDirectory() as tmp_dir:
        comp_content_path = str(Path(tmp_dir) / "comp" / "content")

        original_cwd = os.getcwd()
        os.chdir(tmp_dir)
        try:
            comp_path, blog_path = generator.save_with_dual_write(
                content=content,
                competition_content_path=comp_content_path,
                filename=filename,
                title="Test Article",
                description="A test article",
            )

            # Both files should exist
            assert Path(comp_path).exists(), f"Competition file missing: {comp_path}"
            assert Path(blog_path).exists(), f"Blog file missing: {blog_path}"

            # Both files should have identical content (byte-for-byte)
            comp_bytes = Path(comp_path).read_bytes()
            blog_bytes = Path(blog_path).read_bytes()
            assert comp_bytes == blog_bytes, (
                "Competition and blog copies are not byte-identical"
            )
        finally:
            os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Property 38: Blog Index Integrity
# Feature: kaggle-ml-toolkit, Property 38
# ---------------------------------------------------------------------------


@given(
    title=st.text(
        min_size=3, max_size=50,
        alphabet=st.characters(whitelist_categories=("L", "N", "Z")),
    ),
    filename=st.text(
        min_size=3, max_size=20,
        alphabet=st.characters(whitelist_categories=("Ll", "Nd")),
    ).map(lambda x: x + ".md"),
)
@settings(max_examples=20)
def test_blog_index_integrity(title, filename):
    """After save_with_dual_write, the blog index contains an entry
    for the saved article.

    **Validates: Requirements 25.3**
    """
    generator = ContentGenerator()

    with tempfile.TemporaryDirectory() as tmp_dir:
        comp_content_path = str(Path(tmp_dir) / "comp" / "content")

        original_cwd = os.getcwd()
        os.chdir(tmp_dir)
        try:
            generator.save_with_dual_write(
                content="# Test Content\nSome text.",
                competition_content_path=comp_content_path,
                filename=filename,
                title=title,
                description="Test description",
            )

            # Blog index should exist and contain the entry
            index_path = Path("docs/blog/README.md")
            assert index_path.exists(), "Blog index README.md not created"

            index_content = index_path.read_text(encoding="utf-8")
            assert filename in index_content, (
                f"Filename '{filename}' not found in blog index"
            )
            assert title in index_content, (
                f"Title '{title}' not found in blog index"
            )
        finally:
            os.chdir(original_cwd)
