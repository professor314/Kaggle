# Research Document: Titanic Survival Prediction

## Domain Background

### The Disaster

On April 15, 1912, the RMS Titanic sank after colliding with an iceberg during her maiden voyage from Southampton to New York City. Of the estimated 2,224 passengers and crew aboard, more than 1,500 died, making it one of the deadliest peacetime maritime disasters in history.

### Passenger Demographics and Class System

The Titanic carried three classes of passengers reflecting the rigid social hierarchy of the era:

- **First Class** (~324 passengers): Wealthy elites, industrialists, and aristocrats. Cabins located on upper decks (A through C), closest to the boat deck where lifeboats were stationed.
- **Second Class** (~284 passengers): Middle-class professionals, clergy, and tourists. Cabins on middle decks (D through F).
- **Third Class** (~709 passengers): Immigrants and working-class travelers. Cabins on lower decks (F and G), farthest from lifeboats with more restricted access routes to the boat deck.

### Survival Rates by Group

Historical records reveal stark survival disparities:

| Group | Survival Rate |
|-------|--------------|
| First Class Women | ~97% |
| Second Class Women | ~86% |
| Third Class Women | ~49% |
| First Class Men | ~33% |
| Second Class Men | ~8% |
| Third Class Men | ~16% |
| Children (1st/2nd) | ~100% |
| Children (3rd) | ~34% |
| Crew | ~24% |

The "women and children first" protocol was enforced, though unevenly. Officers on the port side interpreted the rule strictly (women and children only), while starboard-side officers allowed men if no women were waiting.

### Ship Layout Relevant to Evacuation

- Lifeboats were located on the Boat Deck (top deck). First-class passengers had the most direct access.
- Third-class passengers faced physical barriers (locked gates in some cases), language barriers (many were non-English-speaking immigrants), and longer travel distances to reach the boat deck.
- The ship carried only 20 lifeboats (capacity ~1,178), far fewer than needed for all passengers.
- Cabin deck letters (A-G) directly correlate with vertical position on the ship and thus evacuation time.

## Prior Work Summary

### Known Top Approaches from Kaggle Titanic Competition

The Titanic competition is Kaggle's most popular Getting Started competition with thousands of submissions and public kernels providing extensive benchmarking data.

#### Baseline Approaches

- **Gender-only model**: Predicting all females survive, all males die achieves ~77% accuracy. This is the simplest meaningful baseline.
- **Gender + Class model**: Adding passenger class improves to ~78-79%.

#### Mid-Range Approaches (78-82%)

- **Tree-based models** (Random Forest, Gradient Boosting) with basic features typically achieve 78-82% accuracy.
- **Logistic Regression** with engineered features can reach similar performance (~78-80%).
- Key features at this level: Sex, Pclass, Age (imputed), Fare, Embarked.

#### Top Solutions (83-85%)

Top-performing solutions consistently share these characteristics:

1. **Extensive feature engineering**:
   - Title extraction from Name (Mr, Mrs, Miss, Master, rare titles)
   - Family grouping (SibSp + Parch combinations, surname-based family identification)
   - Deck assignment from Cabin letter
   - Fare per person (Fare / family size)
   - Age imputation using title-based median rather than global median

2. **Model choices**:
   - XGBoost and LightGBM with careful hyperparameter tuning
   - Ensemble methods (stacking, blending) combining multiple model types
   - Random Forest with optimized depth and feature sampling

3. **Data handling**:
   - Cross-validation with stratified folds (essential for imbalanced-ish 38% survival rate)
   - Careful handling of missing data (Age: ~20% missing, Cabin: ~77% missing, Embarked: 2 missing)

#### Notable Public Kernels and Discussions

- "Introduction to Ensembling/Stacking" by Anisotropic: demonstrates stacking approach reaching 82%+
- "Titanic Data Science Solutions" by Manav Sehgal: comprehensive EDA and feature engineering walkthrough
- "A Statistical Analysis & ML workflow" by Masumrumi: systematic approach with 84% accuracy

## Key Hypotheses

### H1: Gender is the Strongest Single Predictor
Women survived at dramatically higher rates due to the "women and children first" evacuation protocol. This is expected to be the single most important feature.

### H2: Passenger Class Strongly Predicts Survival
First-class passengers had:
- Physical proximity to lifeboats (upper decks)
- Better access and fewer barriers to the boat deck
- Possible preferential treatment by crew
- Greater awareness of the danger (better communication)

### H3: Age Affects Survival, Especially for Children
Children were prioritized in evacuation. Very young children had higher survival across all classes. Elderly passengers may have had lower survival due to mobility constraints.

### H4: Family Size Has a Non-Linear Effect on Survival
- Solo travelers (SibSp=0, Parch=0) had lower survival—no one to help them or wait for them
- Small families (2-4 members) had higher survival—mutual support during evacuation
- Very large families (5+) had lower survival—difficulty evacuating together, unwillingness to leave family members behind

### H5: Fare and Cabin Are Proxies for Socioeconomic Status
- Higher fares correlate with better cabins (upper decks)
- Cabin letter indicates deck level and thus evacuation access
- Fare captures within-class variation (better cabin placement)

### H6: Title Extracted from Name Reveals Social Status and Demographics
Names in the format "Surname, Title. Firstname" contain titles that indicate:
- Gender (Mr vs Mrs/Miss)
- Age (Master = boy, Miss = unmarried/young woman)
- Social status (Dr, Rev, military titles, nobility titles like Countess, Sir)
- Marital status (Mrs vs Miss)

## Domain Knowledge for Feature Engineering

### Name and Title Analysis
- Format: "Last, Title. First Middle"
- Common titles: Mr, Mrs, Miss, Master
- Rare titles indicating status: Dr, Rev, Col, Major, Capt, Sir, Lady, Countess, Don, Dona, Jonkheer
- "Master" specifically indicates a male child (reliable age proxy when Age is missing)
- Rare titles can be grouped into categories: Royalty, Officer, Professional

### Cabin and Deck Information
- Cabin format: DeckNumber (e.g., "C85" = Deck C, cabin 85)
- Deck hierarchy (top to bottom): A, B, C, D, E, F, G, T
- ~77% of cabin values are missing—missingness itself may be informative (lower-class passengers less likely to have recorded cabins)
- When present, cabin letter is a strong signal for proximity to lifeboats

### Embarkation Port
- **C** = Cherbourg (France): Higher proportion of first-class passengers
- **Q** = Queenstown (Ireland): Higher proportion of third-class passengers (emigrants)
- **S** = Southampton (England): Mixed, largest group
- Port correlates with class and nationality, making it a secondary indicator

### Family Size Computation
- `FamilySize = SibSp + Parch + 1` (including self)
- `IsAlone = 1 if FamilySize == 1 else 0`
- Optimal family size for survival appears to be 2-4 based on prior analyses

### Fare Analysis
- Fare reflects ticket price, sometimes shared among family/group members
- `FarePerPerson = Fare / FamilySize` is a more accurate wealth proxy
- Fare bins: Low (0-7.91), Mid-Low (7.91-14.45), Mid-High (14.45-31), High (31+) based on quartiles

### Missing Data Patterns
| Feature | Missing % | Recommended Strategy |
|---------|-----------|---------------------|
| Age | ~20% | Impute using Title-group median |
| Cabin | ~77% | Extract deck letter where available; create "has_cabin" flag |
| Embarked | <1% | Fill with mode ('S') |
| Fare | <1% | Fill with class-specific median |

## Recommended Approaches

### Feature Engineering Pipeline

1. **Extract Title** from Name using regex pattern ` ([A-Za-z]+)\.`
2. **Group rare titles** into categories (Royalty, Officer, Professional, Common)
3. **Create FamilySize** = SibSp + Parch + 1
4. **Create IsAlone** flag (FamilySize == 1)
5. **Extract Deck** from Cabin (first character) where available
6. **Bin Age** into groups (Child: 0-12, Teen: 13-17, Young Adult: 18-35, Adult: 36-55, Senior: 55+)
7. **Bin Fare** into quartile-based groups
8. **Compute FarePerPerson** = Fare / FamilySize

### Modeling Strategy

1. **Baseline**: Gender-only prediction (~77%)
2. **Logistic Regression**: Interpretable baseline with engineered features
3. **Random Forest**: Strong tree-based model, good with mixed feature types
4. **Gradient Boosting (XGBoost/LightGBM)**: Typically top performer for tabular data
5. **Ensemble**: Stack or blend top 2-3 models for final submission

### Evaluation Strategy

- **Primary metric**: Accuracy (competition default)
- **Secondary metrics**: Precision, Recall, F1 by class (to understand error patterns)
- **Validation**: Stratified 5-fold cross-validation (maintain ~38% survival rate in each fold)
- **Holdout**: Consider 80/20 stratified train/validation split for final model selection

### Expected Performance Targets

| Approach | Expected Accuracy |
|----------|------------------|
| Gender baseline | ~77% |
| LR + basic features | ~78-79% |
| RF + engineered features | ~80-82% |
| GBM + full feature set | ~81-83% |
| Optimized ensemble | ~83-85% |
