"""Content Generator module for producing educational blog posts, Kaggle writeups, and articles.

Generates structured markdown content from project artifacts, supporting multiple
audience levels and dual-save to both competition and blog directories.
"""

from typing import List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import os


class ContentGenerator:
    """Structured blog post, Kaggle writeup, and article generation.

    Produces educational content from competition artifacts including research
    documents, design docs, EDA output, and model results. Supports dual-save
    to competition content directories and the shared docs/blog/ directory.
    """

    def generate_blog_post(
        self,
        competition_dir: str,
        target_audience: str = "intermediate",
        include_code: bool = True,
    ) -> str:
        """Generate a full blog post draft from project artifacts.

        Reads available artifacts from the competition directory and produces
        a structured markdown blog post covering the full ML pipeline journey.

        Args:
            competition_dir: Path to the competition directory containing
                research/, content/, and other artifact folders.
            target_audience: One of "beginner", "intermediate", "advanced".
                Controls technical depth of explanations.
            include_code: Whether to include code snippets in the post.

        Returns:
            Markdown string containing the complete blog post.
        """
        comp_path = Path(competition_dir)

        # Read available artifacts
        research_content = self._read_artifact(
            comp_path / "research" / "research_document.md"
        )
        design_content = self._read_artifact(
            comp_path / "research" / "design_doc.md"
        )
        existing_content = self._read_content_folder(comp_path / "content")

        # Determine audience-specific depth
        depth = self._get_audience_depth(target_audience)

        # Build the blog post sections
        sections = []

        # Title
        comp_name = comp_path.name.replace("_", " ").replace("-", " ").title()
        sections.append(f"# {comp_name}: A Complete Machine Learning Walkthrough\n")

        # Introduction
        sections.append(self._generate_introduction(
            comp_name, research_content, depth
        ))

        # Domain Research Summary
        sections.append(self._generate_research_section(
            research_content, design_content, depth
        ))

        # EDA Findings
        sections.append(self._generate_eda_section(
            existing_content, depth, include_code
        ))

        # Feature Engineering Rationale
        sections.append(self._generate_feature_engineering_section(
            design_content, depth, include_code
        ))

        # Model Selection Reasoning
        sections.append(self._generate_model_selection_section(
            design_content, existing_content, depth, include_code
        ))

        # Optimization Results
        sections.append(self._generate_optimization_section(
            existing_content, depth, include_code
        ))

        # Final Performance and Lessons Learned
        sections.append(self._generate_conclusion_section(
            existing_content, depth
        ))

        return "\n".join(sections)

    def generate_kaggle_writeup(
        self,
        competition_dir: str,
        final_score: float,
        baseline_score: float,
        ai_tools_used: Optional[List[str]] = None,
    ) -> str:
        """Generate a Kaggle Discussion-format solution writeup.

        Produces a concise, code-focused writeup suitable for posting to
        Kaggle's Discussion forum. Includes AI transparency disclosure.

        Args:
            competition_dir: Path to competition directory.
            final_score: Final leaderboard score achieved.
            baseline_score: Baseline model score for comparison.
            ai_tools_used: List of AI tools used. Defaults to ["Kiro"].

        Returns:
            Path to the saved writeup file.

        Raises:
            FileNotFoundError: If competition_dir does not exist.
        """
        comp_path = Path(competition_dir)
        if not comp_path.exists():
            raise FileNotFoundError(
                f"Competition directory does not exist: {competition_dir}"
            )

        if ai_tools_used is None:
            ai_tools_used = ["Kiro"]

        # Read available artifacts for context
        research_content = self._read_artifact(
            comp_path / "research" / "research_document.md"
        )
        design_content = self._read_artifact(
            comp_path / "research" / "design_doc.md"
        )

        comp_name = comp_path.name.replace("_", " ").replace("-", " ").title()
        improvement = final_score - baseline_score

        # Build writeup sections
        sections = []

        # Title
        sections.append(f"# {comp_name} — Solution Writeup\n")

        # Summary
        sections.append("## Summary\n")
        sections.append(
            f"This solution achieves a score of **{final_score:.4f}**, "
            f"an improvement of **{improvement:+.4f}** over the baseline score "
            f"of {baseline_score:.4f}. "
            f"The approach combines domain research, careful feature engineering, "
            f"and model optimization to achieve competitive performance.\n"
        )

        # Approach
        sections.append("## Approach\n")
        if design_content:
            sections.append(
                "The approach was guided by domain research and structured "
                "decision-making at each pipeline stage:\n"
            )
            sections.append("1. Domain research and hypothesis formation")
            sections.append("2. Exploratory data analysis")
            sections.append("3. Feature engineering informed by domain knowledge")
            sections.append("4. Model selection via cross-validated comparison")
            sections.append("5. Hyperparameter optimization")
            sections.append("6. Final model training and submission\n")
        else:
            sections.append(
                "A systematic approach was used covering data cleaning, "
                "feature engineering, model selection, and optimization.\n"
            )

        # What Worked
        sections.append("## What Worked\n")
        sections.append("- Domain-informed feature engineering")
        sections.append("- Cross-validated model comparison before committing to a final model")
        sections.append("- Structured research phase before modeling")
        sections.append("- Iterative optimization with experiment tracking\n")

        # What Didn't Work
        sections.append("## What Didn't Work\n")
        sections.append("- Some engineered features did not improve performance")
        sections.append("- Overly complex models showed signs of overfitting")
        sections.append("- Initial feature selection was too aggressive\n")

        # Final Model Description
        sections.append("## Final Model Description\n")
        sections.append(
            "The final model was selected based on cross-validated performance "
            "and optimized via hyperparameter search. See the research and design "
            "documents for full decision rationale.\n"
        )

        # Score
        sections.append("## Score\n")
        sections.append(f"| Metric | Score |")
        sections.append(f"|--------|-------|")
        sections.append(f"| Final Score | {final_score:.4f} |")
        sections.append(f"| Baseline Score | {baseline_score:.4f} |")
        sections.append(f"| Improvement | {improvement:+.4f} |\n")

        # AI Disclosure
        sections.append("## AI Disclosure\n")
        sections.append(
            "This solution was developed with AI-assisted tools. "
            "Kaggle explicitly allows AI-assisted development in competitions.\n"
        )
        sections.append("**Tools used:**\n")
        for tool in ai_tools_used:
            sections.append(f"- {tool}")
        sections.append("")

        content = "\n".join(sections)

        # Save to competition content directory
        content_dir = comp_path / "content"
        content_dir.mkdir(parents=True, exist_ok=True)
        filename = "kaggle_writeup.md"

        # Use dual-save to write to both locations
        comp_save_path, blog_save_path = self.save_with_dual_write(
            content=content,
            competition_content_path=str(content_dir),
            filename=filename,
            title=f"{comp_name} — Kaggle Writeup",
            description=f"Solution writeup for {comp_name} competition (score: {final_score:.4f})",
        )

        return comp_save_path

    def generate_meta_article(
        self,
        topic: str,
        output_dir: str = "docs/blog/",
    ) -> str:
        """Generate a meta-article about the specified topic.

        Creates an article about the toolkit itself or related topics,
        saves to the output directory, and updates the blog index.

        Args:
            topic: The topic to write about (e.g., "toolkit-intro",
                "kaggle-cli-setup", "submission-guide").
            output_dir: Directory to save the article. Defaults to "docs/blog/".

        Returns:
            Path to the saved article file.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename from topic
        filename = f"{topic.lower().replace(' ', '-').replace('_', '-')}.md"
        title = topic.replace("-", " ").replace("_", " ").title()
        date = datetime.now().strftime("%Y-%m-%d")

        # Generate article content
        content = self._generate_meta_content(topic, title, date)

        # Save to output directory
        file_path = output_path / filename
        file_path.write_text(content, encoding="utf-8")

        # Update blog index
        self.update_blog_index(
            title=title,
            date=date,
            description=f"Article about {topic.replace('-', ' ').replace('_', ' ')}",
            filename=filename,
        )

        return str(file_path)

    def save_with_dual_write(
        self,
        content: str,
        competition_content_path: str,
        filename: str,
        title: str,
        description: str,
    ) -> Tuple[str, str]:
        """Save content to both competition directory and docs/blog/.

        Writes identical content to two locations and updates the blog index.

        Args:
            content: The markdown content to save.
            competition_content_path: Path to the competition's content directory.
            filename: Name of the file to save.
            title: Title for the blog index entry.
            description: Description for the blog index entry.

        Returns:
            Tuple of (competition_path, blog_path) as strings.
        """
        # Save to competition content directory
        comp_path = Path(competition_content_path)
        comp_path.mkdir(parents=True, exist_ok=True)
        comp_file = comp_path / filename
        comp_file.write_text(content, encoding="utf-8")

        # Save to docs/blog/ directory
        blog_path = Path("docs/blog")
        blog_path.mkdir(parents=True, exist_ok=True)
        blog_file = blog_path / filename
        blog_file.write_text(content, encoding="utf-8")

        # Update blog index
        date = datetime.now().strftime("%Y-%m-%d")
        self.update_blog_index(
            title=title,
            date=date,
            description=description,
            filename=filename,
        )

        return str(comp_file), str(blog_file)

    def update_blog_index(
        self,
        title: str,
        date: str,
        description: str,
        filename: str,
    ) -> None:
        """Add an entry to the docs/blog/README.md index file.

        Creates the index file with a header and table if it doesn't exist.
        Appends a new row to the table for the article.

        Args:
            title: Article title.
            date: Publication date (YYYY-MM-DD format).
            description: Brief description of the article.
            filename: Filename of the article in docs/blog/.
        """
        index_path = Path("docs/blog/README.md")
        index_path.parent.mkdir(parents=True, exist_ok=True)

        table_header = "| Title | Date | Description |\n| ----- | ---- | ----------- |"
        new_row = f"| [{title}](./{filename}) | {date} | {description} |"

        if not index_path.exists():
            # Create new index file with header and table
            content = (
                "# Blog Articles\n\n"
                "A collection of articles, writeups, and educational content "
                "from the Kaggle ML Toolkit project.\n\n"
                f"{table_header}\n"
                f"{new_row}\n"
            )
            index_path.write_text(content, encoding="utf-8")
        else:
            existing = index_path.read_text(encoding="utf-8")

            if "| Title | Date | Description |" in existing:
                # Table exists, append new row
                existing = existing.rstrip("\n") + "\n" + new_row + "\n"
                index_path.write_text(existing, encoding="utf-8")
            else:
                # File exists but no table — append table with entry
                existing = existing.rstrip("\n") + "\n\n" + table_header + "\n" + new_row + "\n"
                index_path.write_text(existing, encoding="utf-8")

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------

    def _read_artifact(self, path: Path) -> str:
        """Read a file if it exists, return empty string otherwise."""
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def _read_content_folder(self, content_dir: Path) -> str:
        """Read all markdown files in a content folder."""
        if not content_dir.exists():
            return ""
        parts = []
        for f in sorted(content_dir.iterdir()):
            if f.suffix == ".md":
                parts.append(f.read_text(encoding="utf-8"))
        return "\n".join(parts)

    def _get_audience_depth(self, target_audience: str) -> str:
        """Map audience level to depth descriptor."""
        mapping = {
            "beginner": "detailed",
            "intermediate": "balanced",
            "advanced": "technical",
        }
        return mapping.get(target_audience, "balanced")

    def _generate_introduction(
        self, comp_name: str, research_content: str, depth: str
    ) -> str:
        """Generate the introduction section."""
        section = "## Introduction\n\n"

        if depth == "detailed":
            section += (
                f"This post walks through a complete machine learning project "
                f"for the {comp_name} competition. We'll explain every step in "
                f"detail, from understanding the problem to submitting predictions. "
                f"Whether you're new to machine learning or looking for a structured "
                f"approach, this walkthrough covers the full journey.\n"
            )
        elif depth == "technical":
            section += (
                f"A technical walkthrough of the {comp_name} competition, "
                f"covering research-driven feature engineering, model selection, "
                f"and optimization strategies.\n"
            )
        else:
            section += (
                f"This post documents the approach taken for the {comp_name} "
                f"competition, covering the full pipeline from domain research "
                f"through model optimization to final submission.\n"
            )

        if research_content:
            section += (
                "\nThe approach was guided by domain research conducted before "
                "any modeling began, ensuring decisions were informed by "
                "domain expertise and prior work.\n"
            )

        return section

    def _generate_research_section(
        self, research_content: str, design_content: str, depth: str
    ) -> str:
        """Generate the domain research summary section."""
        section = "## Domain Research Summary\n\n"

        if research_content:
            if depth == "detailed":
                section += (
                    "Before writing any modeling code, we conducted thorough "
                    "domain research. This research-first approach ensures that "
                    "modeling decisions are grounded in understanding of the problem "
                    "domain rather than blind experimentation.\n\n"
                )
            section += (
                "Key findings from the research phase informed our feature "
                "engineering and model selection decisions. The research document "
                "captured domain background, prior work, hypotheses, and "
                "recommended approaches.\n"
            )
        else:
            section += (
                "No formal research document was found for this competition. "
                "A research-first approach is recommended for future projects.\n"
            )

        if design_content:
            section += (
                "\nThe design document recorded strategic decisions made based "
                "on research findings, including primary hypothesis, modeling "
                "strategy, and evaluation priorities.\n"
            )

        return section

    def _generate_eda_section(
        self, existing_content: str, depth: str, include_code: bool
    ) -> str:
        """Generate the EDA findings section."""
        section = "## Exploratory Data Analysis Findings\n\n"

        if depth == "detailed":
            section += (
                "Exploratory Data Analysis (EDA) is the process of examining "
                "your data to understand its structure, patterns, and potential "
                "issues before modeling. Here's what we found:\n\n"
            )
        elif depth == "technical":
            section += "Key statistical findings from the EDA phase:\n\n"
        else:
            section += "The EDA phase revealed several important patterns:\n\n"

        section += "- Distribution analysis of all features\n"
        section += "- Correlation patterns between features and target\n"
        section += "- Missing value patterns and imputation strategies\n"
        section += "- Outlier identification and handling decisions\n"

        if include_code:
            section += (
                "\n```python\n"
                "from kaggle_ml_toolkit.eda_engine import EDAEngine\n\n"
                "eda = EDAEngine()\n"
                "results = eda.run(df, target_column='target')\n"
                "```\n"
            )

        return section

    def _generate_feature_engineering_section(
        self, design_content: str, depth: str, include_code: bool
    ) -> str:
        """Generate the feature engineering rationale section."""
        section = "## Feature Engineering Rationale\n\n"

        if depth == "detailed":
            section += (
                "Feature engineering is the process of creating new input "
                "variables from existing data to help models learn better "
                "patterns. Our feature engineering was guided by domain "
                "knowledge from the research phase.\n\n"
            )
        else:
            section += (
                "Feature engineering decisions were informed by domain research "
                "and EDA findings:\n\n"
            )

        section += "- Interaction features between correlated variables\n"
        section += "- Domain-specific feature creation based on research\n"
        section += "- Binning of continuous variables where appropriate\n"
        section += "- Feature selection to reduce dimensionality\n"

        if include_code:
            section += (
                "\n```python\n"
                "from kaggle_ml_toolkit.feature_engineer import FeatureEngineer\n\n"
                "fe = FeatureEngineer()\n"
                "df = fe.create_interaction(df, 'feature_a', 'feature_b')\n"
                "```\n"
            )

        return section

    def _generate_model_selection_section(
        self, design_content: str, existing_content: str, depth: str, include_code: bool
    ) -> str:
        """Generate the model selection reasoning section."""
        section = "## Model Selection Reasoning\n\n"

        if depth == "detailed":
            section += (
                "Rather than picking a single model, we compared multiple "
                "candidates using cross-validation. This gives us confidence "
                "that our choice is robust and not just lucky on one split.\n\n"
            )
        elif depth == "technical":
            section += (
                "Multi-model comparison via stratified k-fold cross-validation:\n\n"
            )
        else:
            section += (
                "We compared multiple model candidates using cross-validated "
                "evaluation to identify the best approach:\n\n"
            )

        section += "- Logistic Regression (baseline)\n"
        section += "- Random Forest\n"
        section += "- Gradient Boosting\n"
        section += "- Support Vector Machine\n"

        if include_code:
            section += (
                "\n```python\n"
                "from kaggle_ml_toolkit.model_selector import ModelSelector\n\n"
                "selector = ModelSelector()\n"
                "results = selector.compare(X, y, problem_type='classification')\n"
                "```\n"
            )

        return section

    def _generate_optimization_section(
        self, existing_content: str, depth: str, include_code: bool
    ) -> str:
        """Generate the optimization results section."""
        section = "## Optimization Results\n\n"

        if depth == "detailed":
            section += (
                "Hyperparameter optimization fine-tunes a model's settings "
                "to squeeze out better performance. We used randomized search "
                "to efficiently explore the parameter space.\n\n"
            )
        else:
            section += (
                "Hyperparameter optimization was applied to the top-performing "
                "model candidates:\n\n"
            )

        section += "- Parameter search method and space explored\n"
        section += "- Cross-validated scores during optimization\n"
        section += "- Final best parameters selected\n"
        section += "- Performance improvement over default parameters\n"

        if include_code:
            section += (
                "\n```python\n"
                "from kaggle_ml_toolkit.model_optimizer import ModelOptimizer\n\n"
                "optimizer = ModelOptimizer()\n"
                "result = optimizer.optimize(model, param_grid, X, y)\n"
                "```\n"
            )

        return section

    def _generate_conclusion_section(
        self, existing_content: str, depth: str
    ) -> str:
        """Generate the final performance and lessons learned section."""
        section = "## Final Performance and Lessons Learned\n\n"

        section += "### Results\n\n"
        section += (
            "The final model was evaluated on the held-out test set and "
            "submitted to the competition leaderboard.\n\n"
        )

        section += "### What Worked\n\n"
        section += "- Research-first approach informed better feature engineering\n"
        section += "- Systematic model comparison avoided premature commitment\n"
        section += "- Cross-validation provided reliable performance estimates\n\n"

        section += "### What Didn't Work\n\n"
        section += "- Some hypothesized features did not improve performance\n"
        section += "- Initial models were overfit before proper CV was applied\n\n"

        section += "### Lessons Learned\n\n"
        if depth == "detailed":
            section += (
                "- Always start with domain research before modeling\n"
                "- Simple baselines provide essential reference points\n"
                "- Cross-validation is crucial for reliable evaluation\n"
                "- Document decisions for future reference and learning\n"
                "- Feature engineering guided by domain knowledge outperforms "
                "blind feature creation\n"
            )
        else:
            section += (
                "- Domain research before modeling pays dividends\n"
                "- Structured decision-making improves reproducibility\n"
                "- Documentation enables educational content creation\n"
            )

        return section

    def _generate_meta_content(
        self, topic: str, title: str, date: str
    ) -> str:
        """Generate meta-article content based on topic."""
        content = f"# {title}\n\n"
        content += f"*Published: {date}*\n\n"

        topic_lower = topic.lower().replace("_", "-")

        if "intro" in topic_lower or "toolkit" in topic_lower:
            content += (
                "## What is the Kaggle ML Toolkit?\n\n"
                "The Kaggle ML Toolkit is an AI-assisted research and learning "
                "platform for Kaggle competitions. It provides a structured workflow "
                "where you collaborate with Kiro to make informed decisions at every "
                "stage of a machine learning project.\n\n"
                "## Design Philosophy\n\n"
                "1. **Research-first**: Domain knowledge before modeling\n"
                "2. **Decision-driven**: Distilled information for strategic choices\n"
                "3. **Educational output**: Every project produces publishable content\n"
                "4. **Phased delivery**: Incremental feature development\n"
                "5. **Explainability**: Every decision documented and justified\n\n"
                "## Getting Started\n\n"
                "Install the toolkit and run your first competition pipeline "
                "using the provided steering files and Jupyter notebook templates.\n"
            )
        elif "cli" in topic_lower or "setup" in topic_lower:
            content += (
                "## Setting Up the Kaggle CLI\n\n"
                "The Kaggle CLI provides command-line access to download datasets, "
                "submit predictions, and manage competitions.\n\n"
                "## Installation\n\n"
                "```bash\npip install kaggle\n```\n\n"
                "## Configuration\n\n"
                "1. Go to kaggle.com → Account → API → Create New Token\n"
                "2. Save `kaggle.json` to `~/.kaggle/`\n"
                "3. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`\n\n"
                "## Common Commands\n\n"
                "```bash\n"
                "kaggle competitions list\n"
                "kaggle competitions download -c titanic\n"
                "kaggle competitions submit -c titanic -f submission.csv -m \"My submission\"\n"
                "```\n"
            )
        elif "submission" in topic_lower or "guide" in topic_lower:
            content += (
                "## Submitting to Kaggle\n\n"
                "There are two submission types on Kaggle:\n\n"
                "### CSV Submission\n\n"
                "Upload a predictions file with the required columns (usually an ID "
                "and a prediction column).\n\n"
                "### Code Competition\n\n"
                "Submit a notebook that runs on Kaggle's servers. The notebook reads "
                "from `/kaggle/input/` and writes to `/kaggle/working/`.\n\n"
                "## Tips\n\n"
                "- You typically get 10 submissions per day\n"
                "- Use cross-validation locally before submitting\n"
                "- Track your submissions with experiment IDs\n"
            )
        elif "phase" in topic_lower or "development" in topic_lower:
            content += (
                "## Phased Development Approach\n\n"
                "The toolkit is built incrementally across three phases:\n\n"
                "### Phase 1: Core Pipeline\n\n"
                "Data loading, cleaning, feature engineering, model selection, "
                "evaluation, submission, domain research, EDA, and content generation.\n\n"
                "### Phase 2: Advanced Features\n\n"
                "Ensembles, interpretability (SHAP), augmentation, advanced CV, "
                "Kaggle CLI automation, and Code Competition support.\n\n"
                "### Phase 3: Deep Learning\n\n"
                "PyTorch/TensorFlow integration, cloud scaling (AWS), and "
                "advanced experiment tracking.\n\n"
                "## Why Phased?\n\n"
                "Incremental development ensures each feature is tested and "
                "documented before moving on. This approach also makes the "
                "project accessible to contributors at any stage.\n"
            )
        else:
            content += (
                f"## About {title}\n\n"
                f"This article covers {topic.replace('-', ' ').replace('_', ' ')} "
                f"as part of the Kaggle ML Toolkit project.\n\n"
                "## Overview\n\n"
                "Content for this topic will be expanded based on project progress "
                "and user feedback.\n"
            )

        return content
