# Kaggle Platform Guide

**Last Updated:** July 2025

This guide covers the key mechanics of the Kaggle competition platform — competition types, medals, progression, submission workflows, and policies relevant to AI-assisted development.

---

## Competition Types

### Getting Started

Beginner-friendly competitions designed to teach the fundamentals of the Kaggle workflow. Examples include Titanic (classification), House Prices (regression), and Spaceship Titanic (classification).

- **Do NOT award medals or points**
- Remain open indefinitely (no deadline)
- Great for learning data loading, EDA, model training, and submission mechanics
- Recommended starting point before tackling medal-eligible competitions

### Playground

Monthly competitions using synthetic or curated datasets. These rotate regularly and provide a low-pressure environment to practice.

- **Award medals**
- Typically run for 2–4 weeks
- Good for practicing new techniques and building medal count
- Datasets are often synthetic, so domain expertise matters less

### Featured

Major sponsored competitions with cash prizes and high visibility. These represent the competitive core of Kaggle.

- **Award medals and points**
- Prize pools range from $10,000 to $1,000,000+
- Run for 2–3 months typically
- Attract top competitors worldwide
- Often require significant domain expertise and compute resources

### Research

Academic and scientific competitions focused on advancing knowledge in specific domains.

- **Award medals**
- Often partnered with research institutions or government agencies
- May have unique evaluation criteria tied to scientific goals
- Datasets may involve specialized domains (genomics, climate, etc.)

### Community

Competitions created by Kaggle users rather than corporate sponsors.

- **May or may not award medals** (depends on how the competition is configured)
- Quality and difficulty vary widely
- Can be a good way to practice niche topics

---

## Medal System

Medals are awarded based on your team's final ranking on the **private leaderboard** (not the public leaderboard shown during the competition).

| Medal | Requirement |
|--------|------------|
| Bronze | Top 40% of teams |
| Silver | Top 20% of teams |
| Gold   | Top 10% of teams (standard) |

### Gold Medal Scaling for Large Competitions

For competitions with many teams, the Gold medal threshold uses a scaled formula:

- **Top 10 teams + 0.2% of remaining teams** receive Gold
- This means 1 extra Gold medal slot per 500 additional teams beyond the base 10
- Example: A competition with 5,000 teams would award Gold to roughly the top 20 teams

### Important Notes

- **Getting Started competitions do NOT award medals** regardless of placement
- Medals are determined by the **private leaderboard** (final rankings), not the public leaderboard
- Only your best-scoring selected submissions count toward final ranking

---

## Progression Tiers

Kaggle ranks users through a tier system based on medal accumulation:

| Tier | Requirements |
|------|-------------|
| **Novice** | New account, no activity |
| **Contributor** | Complete profile tasks (bio, run a notebook, make a submission, upvote, participate in a discussion) |
| **Expert** | 2 Bronze medals |
| **Master** | 1 Gold + 2 Silver medals |
| **Grandmaster** | 5 Gold medals (with at least 1 solo Gold) |

Each category (Competitions, Datasets, Notebooks, Discussion) has its own tier progression. The requirements above are for Competitions specifically.

---

## Submission Types

### Simple Submission (CSV Upload)

The traditional submission format where you generate predictions locally and upload a CSV file.

**Workflow:**
1. Download competition data (via web UI or `kaggle competitions download`)
2. Train your model locally (or in a Kaggle notebook)
3. Generate predictions on the test set
4. Format predictions as a CSV with the required columns (typically an ID column + prediction column)
5. Upload via the Kaggle website or submit via CLI: `kaggle competitions submit`

**Used by:** Getting Started competitions, many Featured competitions, and most Playground competitions.

### Code Competition

Submissions are Jupyter notebooks that execute on Kaggle's servers. Your notebook must produce the output file during execution.

**Constraints:**
- Notebook reads input data from `/kaggle/input/`
- Notebook writes output to `/kaggle/working/`
- **No internet access** during execution
- Limited compute time (typically 9 hours for GPU, 9 hours for CPU)
- Limited RAM and disk space
- All dependencies must be pre-installed in the Kaggle environment or bundled with the notebook

**Workflow:**
1. Develop and test your notebook on Kaggle (or locally with matching structure)
2. Ensure all data references use `/kaggle/input/{competition-name}/`
3. Write your submission file to `/kaggle/working/submission.csv`
4. Submit the notebook — Kaggle re-runs it from scratch on the private test data

**Used by:** Many Featured competitions with cash prizes, competitions requiring reproducibility guarantees.

---

## AI-Assisted Development Policy

Kaggle explicitly allows AI-assisted development in competitions:

- **AI coding assistants are permitted** — tools like Kiro, GitHub Copilot, ChatGPT, Claude, Gemini, and others are all allowed
- **Teams have won competitions** (including first place) using multiple LLM agents collaboratively
- **No restriction** on the degree of AI assistance used in developing your solution
- The **human competitor is responsible** for understanding and being able to defend their approach
- **Recommendation:** Include transparent AI disclosure in competition writeups and discussion posts — the community values honesty about methodology

The key principle: AI is a tool in your workflow, not a replacement for understanding. You should be able to explain why your approach works.

---

## Submission Rules

### Daily Submission Limit

- **Maximum 10 submissions per day** (typical for most competitions)
- Some competitions may have different limits — always check the competition rules page
- The limit resets at midnight UTC

### Leaderboard Mechanics

Kaggle uses a split evaluation system:

- **Public Leaderboard:** Shows your score on a subset of the test data during the competition. Visible to all participants.
- **Private Leaderboard:** Uses the full test set (or a different subset). Revealed only after the competition ends. Determines final rankings and medal awards.

**Why this matters:** Overfitting to the public leaderboard is a common trap. A model that scores well on the public portion may perform poorly on the private portion if it has overfit to noise rather than learning generalizable patterns.

### Best Practices

- Select your final submissions carefully (usually you choose 2 submissions for final evaluation)
- Trust your cross-validation score over the public leaderboard score
- Monitor the gap between your CV score and public LB score — a large gap suggests overfitting

---

## Team Rules

- **Maximum team size** varies by competition (commonly 5–10 members)
- **Team Merger Deadline:** Teams must merge before this deadline (typically 1–2 weeks before competition end)
- When teams merge, their **combined submission count must not exceed** the maximum daily limit multiplied by competition duration
- All team members share the same medal if the team places in medal range
- One account per person — multi-accounting is strictly prohibited

---

## Tips for New Competitors

1. **Start with Getting Started competitions** to learn the mechanics without pressure. Titanic and House Prices are the classic starting points.

2. **Read the Discussion forum** for each competition — top competitors often share insights, starter notebooks, and data observations.

3. **Study top solutions from past competitions** — after competitions end, winners typically post detailed solution writeups. These are goldmines for learning.

4. **Focus on understanding the problem domain** before jumping into modeling. Domain knowledge often matters more than algorithm choice.

5. **Build a solid cross-validation strategy** early — this is your most reliable feedback signal, more trustworthy than the public leaderboard.

6. **Iterate quickly** — start with a simple baseline, submit it, then improve incrementally. Don't spend weeks on a perfect first attempt.

7. **Join teams** once you're comfortable — you'll learn faster from experienced competitors, and collaboration makes the process more enjoyable.

8. **Track your experiments** — log what you tried, what worked, and what didn't. This practice pays dividends across competitions.
