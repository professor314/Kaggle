---
phase: 8
title: Content Creation
triggers: kaggle_ml_toolkit.content_generator
---

# Content Creation

## Context
Every competition produces publishable content — blog posts, Kaggle writeups, and educational articles. This phase converts our research, decisions, and results into structured narratives. Content creation serves dual purposes: sharing knowledge with the community and reinforcing our own learning through explanation.

## Publishing workflow (default: draft first, publish when finished)
- **Write blog posts as `draft: true` and keep them offline until the project is
  actually done and the user approves.** Do NOT auto-publish a post the moment a
  first result lands. Build it locally, finish it, then flip to `draft: false`,
  deploy, and (if applicable) let the onion mirror pick it up.
- To take a live post down: set `draft: true`, delete its folder from `public/`,
  `aws s3 rm s3://imadestuff.com/posts/<slug> --recursive`, then invalidate
  CloudFront (`/posts/<slug>/*`, `/posts/`, `/`). Keep the `.md` in the repo.

## Interaction style (user preference)
- **Keep moving; don't narrate hedging.** Avoid opening statements with "honestly",
  "to be honest", "the honest truth", or long meta-caveats. State the result and
  the next action plainly and proceed. Record caveats in the experiments_log, not
  as repeated conversational hedging.

## Decision Points

### Target Audience
**Question:** Who is the primary audience for this content?
**Options:**
- Beginner: Explain concepts from first principles, include background context, focus on "why" over "how."
- Intermediate: Assume ML basics, focus on techniques and trade-offs, include code snippets.
- Advanced: Assume strong foundations, focus on novel insights, unusual approaches, and detailed results.
- Mixed: Write for intermediate but include expandable beginner context and advanced footnotes.
**Recommendation:** Intermediate — the largest audience on Kaggle. Add beginner sidebars for accessibility.

### Content Artifacts
**Question:** Which content pieces should we produce?
**Options:**
- Blog post only: Long-form educational article covering the full journey. Best for portfolio.
- Kaggle writeup only: Concise solution summary for Kaggle Discussions. Best for community engagement.
- Both: Blog post for depth, Kaggle writeup for conciseness. Different formats serve different purposes.
- Full suite: Blog + Kaggle writeup + meta-article about the toolkit/process.
**Recommendation:** Both — the blog post is for your portfolio and learning; the writeup is for the Kaggle community.

### Content Format
**Question:** What format should the primary content take?
**Options:**
- Narrative journey: Tell the story chronologically — research → EDA → features → models → results.
- Results-first: Lead with the outcome, then explain how you got there. Better for busy readers.
- Tutorial style: Step-by-step guide that readers can follow along. Best for educational content.
- Comparison essay: Focus on what worked vs. what didn't, with analysis of why.
**Recommendation:** Narrative journey for blog posts (readers follow along); results-first for Kaggle writeups (reviewers scan quickly).

### Lessons to Highlight
**Question:** Which insights should we emphasize in the content?
**Options:**
- What worked: Focus on successful techniques, positive results, winning strategies.
- What didn't work: Emphasize failures, dead ends, and what we learned from them. Often more valuable.
- Balanced: Cover both successes and failures equally. Most honest and educational.
- Surprising findings: Highlight counterintuitive results, unexpected feature importance, or model behavior.
**Recommendation:** Balanced with emphasis on surprising findings — the community values honesty about failures and novel insights.

### AI Disclosure
**Question:** How should we disclose AI tool usage?
**Options:**
- Brief mention: Single line noting AI-assisted development. Minimum Kaggle requirement.
- Detailed section: Describe which AI tools were used and for what parts of the workflow.
- Integrated narrative: Weave AI collaboration into the story as a methodology choice.
**Recommendation:** Detailed section — transparency builds trust, and Kaggle explicitly allows AI tools. Show how human+AI collaboration works.

## Actions
Based on decisions above, Kiro will:
1. Gather all project artifacts (research doc, design doc, EDA output, model results, submission scores)
2. Execute `ContentGenerator.generate_blog_post()` with chosen audience and format
3. Execute `ContentGenerator.generate_kaggle_writeup()` with scores and AI disclosure
4. Execute `ContentGenerator.save_with_dual_write()` to save to competition dir and docs/blog/
5. Execute `ContentGenerator.update_blog_index()` to maintain the article index
6. Present drafts for user review and editing
7. Ask user: "Ready to publish? Any sections to expand or revise?"
8. Record content decisions and publication status in the design doc
