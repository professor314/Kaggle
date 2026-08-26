# Beyond Prediction: Understanding Who Was Aboard the Titanic Through Clustering

*Published: 2025-07-21*
*Competition: Titanic — Machine Learning from Disaster*
*Technique: K-Means Clustering (Unsupervised Learning)*

---

## Introduction

Everyone who's touched a Kaggle dataset has tried to predict Titanic survival. It's the "Hello World" of machine learning — train a model, tune some hyperparameters, submit a CSV. But what if we asked a different question entirely?

Instead of *"who survived?"*, what if we asked *"who were these people?"*

Prediction treats passengers as rows in a matrix. Clustering treats them as human beings who naturally group into archetypes — wealthy elites, young workers, immigrant families, solo adventurers. These groupings emerge from the data itself, without being told what to look for.

This article explores what K-means clustering reveals about the Titanic's passenger population, and how overlaying survival rates on those clusters tells a more human story than any predictive model ever could.

## The Approach

K-means clustering is an unsupervised learning algorithm. You give it data and a number of groups (k), and it finds the best way to partition the data into k clusters where members of each cluster are more similar to each other than to members of other clusters.

The key difference from supervised learning: **we never tell the algorithm about survival**. It groups passengers purely on their demographic and travel characteristics. Then — after the grouping is done — we look at how each group fared.

### Features Used

We cluster on seven features, all numeric and scaled:

```python
features = ['IsFemale', 'Pclass', 'TitleCode', 'Age', 'Fare', 'FamilySize', 'IsAlone']
```

- **IsFemale** — binary gender indicator
- **Pclass** — passenger class (1, 2, or 3)
- **TitleCode** — social title extracted from name (Mr=0, Miss=1, Mrs=2, Master=3, Rare=4)
- **Age** — imputed by title-group median where missing
- **Fare** — ticket price in pounds
- **FamilySize** — SibSp + Parch + 1
- **IsAlone** — traveling solo (FamilySize = 1)

All features are StandardScaler-normalized before clustering to prevent fare (range 0–512) from dominating over binary features.

## Finding the Right Number of Clusters

The first question in any clustering analysis: how many groups? Too few and you merge distinct populations. Too many and you split coherent groups into noise.

We use silhouette scores — a metric that measures how well-separated the clusters are — for k=2 through k=10:

| K | Silhouette Score |
|---|-----------------|
| 2 | 0.3428 |
| 3 | 0.3615 |
| 4 | 0.3445 |
| 5 | 0.3640 |
| 6 | 0.3637 |
| 7 | 0.3977 |
| 8 | 0.4129 |
| 9 | 0.4271 |
| 10 | **0.4449** |

![Silhouette scores and elbow method for K selection](../../competitions/titanic/notebooks/clustering_output/optimal_k_selection.png)

The silhouette score increases steadily, with k=10 producing the best separation (0.4449). Higher k values produce better-defined clusters because the Titanic dataset genuinely contains many distinct sub-populations. A ship with 891 passengers from different countries, classes, and family structures naturally contains more than 3 or 4 "types" of people.

We proceed with **k=10** — the optimal silhouette score that also gives us rich narrative material.

## The Passenger Archetypes

Here's what the algorithm found. Remember: it was never told about survival. It grouped people purely by who they were.

| Cluster | Archetype | Size | Survival | Female% | Class | Age | Fare |
|---------|-----------|------|----------|---------|-------|-----|------|
| 0 | Solo Upper-Class Men | 88 (9.9%) | 27.3% | 1.1% | 1.3 | 43.5 | £29 |
| 1 | Third-Class Solo Men | 308 (34.6%) | 12.0% | 0.0% | 2.8 | 28.6 | £10 |
| 2 | Third-Class Women | 100 (11.2%) | 70.0% | 100% | 2.6 | 23.4 | £21 |
| 3 | Upper-Class Men | 23 (2.6%) | 34.8% | 13.0% | 1.3 | 45.7 | £37 |
| 4 | Large Families (3rd Class) | 34 (3.8%) | 11.8% | 76.5% | 3.0 | 20.3 | £39 |
| 5 | Young Boys | 39 (4.4%) | 59.0% | 0.0% | 2.6 | 4.5 | £34 |
| 6 | Wealthy Mixed-Gender Group | 20 (2.2%) | 70.0% | 60.0% | 1.0 | 31.1 | £279 |
| 7 | Middle-Class Men | 107 (12.0%) | 17.8% | 0.0% | 2.1 | 32.1 | £39 |
| 8 | Solo Third-Class Women | 93 (10.4%) | 72.0% | 100% | 2.6 | 26.5 | £10 |
| 9 | Upper-Class Women | 79 (8.9%) | **96.2%** | 100% | 1.0 | 35.2 | £85 |

![Cluster profiles comparison](../../competitions/titanic/notebooks/clustering_output/cluster_profiles.png)

Let me give these archetypes some life:

### Cluster 9: "The Protected Elite" (96.2% survival)

First-class women, average age 35, paying £85 per ticket. Nearly all survived. These were the wives and daughters of industrialists, diplomats, and old-money families. When the lifeboats launched, they were first in line — and the data shows it. This is the "women and children first" policy working exactly as intended for those with proximity to the boat deck.

### Cluster 8: "Independent Working Women" (72.0% survival)

Solo women in second/third class, average age 26.5, paying £10. Traveling alone, likely servants, governesses, or young women emigrating for work. Despite their lower class, their gender still granted them priority access to lifeboats — though at a lower rate than the wealthy women above.

### Cluster 2: "Third-Class Women with Families" (70.0% survival)

Women from the lower decks, younger (23.4 avg), traveling with family (avg family size 2.7). Their survival rate is remarkably close to the solo women — suggesting that being female mattered more than whether you had children with you. Though 30% still perished, likely those trapped below decks or in the confusion of the final hours.

### Cluster 5: "The Children" (59.0% survival)

Young boys averaging 4.5 years old, traveling with large families (avg 4.5). The "children" part of "women and children first" shows up here — but at only 59%. Many children in large third-class families were lost because entire families stayed together, and large families from steerage struggled to reach the boat deck in time.

### Cluster 4: "Large Immigrant Families" (11.8% survival)

This is the most tragic cluster. Large families (avg 7.3 members!) in third class, predominantly female (76.5%), very young (20.3 avg). Despite being mostly women and children, their survival rate is *lower than solo men in first class*. The combination of a large family, third-class accommodations far from the lifeboats, and language barriers created a death trap. Families couldn't split up — and so families died together.

### Cluster 1: "The Young Male Workers" (12.0% survival)

The single largest cluster: 308 passengers (34.6% of the ship). Third-class men, traveling alone, late twenties, paying £10 for passage. These were the economic migrants — young men seeking opportunity in America. They had everything working against them: male (last priority for lifeboats), third class (farthest from the boats), alone (no one to advocate for them). One in eight survived.

### Cluster 7: "Middle-Class Family Men" (17.8% survival)

Second-class men with families (avg size 2.5). These were the clerks, teachers, and shopkeepers — traveling with wives and children. Their wives likely survived; they did not. A 17.8% survival rate means roughly one in six made it.

### Cluster 0: "Solo Upper-Class Gentlemen" (27.3% survival)

First-class men, older (43.5 avg), traveling alone. The wealthy bachelor industrialists and professionals. Even their money and proximity to the boat deck couldn't override the gender-based evacuation protocol. But 27.3% is still double the rate of third-class men — wealth bought some advantage, even for males.

## Survival Through a Different Lens

Here's what makes clustering powerful. A predictive model would tell you: "being female increases survival probability by X%." Useful, but abstract.

Clustering shows you the *actual groups of humans* and their fates:

**Near-total survival (>70%):**
- Upper-Class Women: 96.2%
- Solo Third-Class Women: 72.0%
- Third-Class Women with Families: 70.0%
- Wealthy Mixed-Gender Group: 70.0%

**Moderate survival (40-60%):**
- Young Boys: 59.0%

**Near-total death (<30%):**
- Solo Upper-Class Gentlemen: 27.3%
- Upper-Class Men (with family): 34.8%
- Middle-Class Family Men: 17.8%
- Third-Class Solo Men: 12.0%
- Large Immigrant Families: 11.8%

The gap between the top and bottom is **84 percentage points**. That's not a statistical abstraction — it's the difference between a first-class woman who stepped calmly into a lifeboat and a young Irish laborer locked below decks.

![PCA visualization showing clusters and survival](../../competitions/titanic/notebooks/clustering_output/pca_clusters_survival.png)

The PCA projection above shows two views of the same data. On the left, colored by cluster membership. On the right, colored by survival. The correspondence is striking — clusters map cleanly onto survival regions, even though the algorithm never saw the survival column.

## What Clustering Reveals That Prediction Doesn't

### 1. The Tragedy of Large Families

A predictive model might tell you "FamilySize > 5 decreases survival." But clustering shows you *who* those large families were: predominantly female, very young, third class. These weren't luxury cruise passengers with nannies — they were immigrant families, likely non-English-speaking, housed deep in steerage. The combination of large group size, physical distance from lifeboats, and unfamiliarity with the ship created an almost-impossible evacuation scenario.

Cluster 4's 11.8% survival rate — despite being 76.5% female — shatters the simplistic "gender predicts survival" narrative.

### 2. Class Stratifies Even Within Gender

Women survived at high rates overall (74% across the dataset). But clustering reveals the stratification:
- Upper-class women: 96.2%
- Solo third-class women: 72.0%
- Third-class women in families: 70.0%
- Women in large immigrant families: 11.8%

The 84-point gap between the best and worst female-majority clusters shows that "women and children first" was not applied equally.

### 3. The Population Structure Itself Is the Story

The largest cluster — 34.6% of all passengers — were solo third-class men. These young workers were the demographic backbone of the ship. They had essentially no chance of survival (12%). The Titanic wasn't primarily carrying wealthy elites; it was carrying a working-class male workforce, supplemented by families seeking new lives and a thin layer of privilege on top.

### 4. The "Ambiguous" Middle

Cluster 5 (Young Boys, 59% survival) represents the decision boundary — the group where the model struggles most. Some of these children were saved; some died with their families. Prediction models treat this as noise. Clustering reveals it as a coherent group whose fate depended on specific circumstances: which lifeboat was nearby, whether their mother was present, which crew member made the call.

## Does Clustering Help Prediction?

We tested this directly. After identifying clusters, we ran a mini-arena comparing four approaches:

| Model | CV Accuracy | vs. Baseline |
|-------|------------|--------------|
| A: Baseline GBM (no clusters) | 0.8507 | — |
| B: GBM + cluster_id feature | 0.8418 | -0.009 |
| C: GBM + cluster_id + OOF survival rate | 0.8474 | -0.003 |
| D: Per-cluster GBM models | 0.8350 | -0.016 |

On this dataset: **no**. Clustering adds insight but not prediction power. The baseline GBM already captures the gender × class × age interactions that define the clusters. With only 891 rows, adding cluster features introduces more noise than signal.

This is actually an important finding. It demonstrates that **unsupervised and supervised learning serve different purposes**:
- Supervised learning optimizes *accuracy* on a known target
- Unsupervised learning reveals *structure* in the population

Both are valuable. They answer different questions.

## Conclusion

The Titanic dataset is always framed as a prediction problem. But prediction tells you *what features matter*. Clustering tells you *who was there*.

When we step back from the prediction task and ask "what natural groups exist in this population?", we discover:
- A ship that was 34.6% young male workers with near-zero survival chances
- A thin elite of first-class women with near-total survival
- A heartbreaking cluster of large immigrant families — mostly women and children — who died at rates worse than solo men of higher class
- A clear hierarchy where class stratified survival even within the "privileged" female gender

These aren't insights you get from a feature importance plot. They're the kind of understanding that turns data into a human story.

The next time you're working on a prediction problem, try running unsupervised learning first. Not to improve your model — but to understand your data. The clusters you find might change how you think about the problem entirely.

---

## Technical Details

**Tools:** Python, scikit-learn (KMeans, PCA, StandardScaler), matplotlib, seaborn
**Features:** 7 numeric features (IsFemale, Pclass, TitleCode, Age, Fare, FamilySize, IsAlone)
**Optimal K:** 10 (silhouette score = 0.4449)
**PCA variance explained:** 35.3% (PC1) + 24.1% (PC2) = 59.4% in 2D projection

**Key code snippet — the clustering pipeline:**

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train[features])

# Find optimal K
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"K={k}: silhouette={score:.4f}")

# Fit with best K
kmeans = KMeans(n_clusters=10, random_state=42, n_init=20)
clusters = kmeans.fit_predict(X_scaled)

# Overlay survival (never seen by the clustering algorithm)
for c in range(10):
    mask = clusters == c
    survival_rate = y_train[mask].mean()
    print(f"Cluster {c}: {mask.sum()} passengers, {survival_rate:.1%} survival")
```

**AI Disclosure:** This analysis was developed with AI assistance (Kiro) for code generation, narrative structure, and iterative refinement. The clustering methodology, feature selection, and interpretation are grounded in standard unsupervised learning practice. All results are reproducible from the provided script.

**Full script:** [`competitions/titanic/notebooks/clustering_analysis.py`](../../competitions/titanic/notebooks/clustering_analysis.py)
