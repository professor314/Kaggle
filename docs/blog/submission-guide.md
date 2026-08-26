# Kaggle Submission Guide

**Published:** 2025-01-20

---

## CSV Format Submissions

Most Kaggle competitions expect a CSV file with an ID column and one or more prediction columns. For example, the Titanic competition expects:

```csv
PassengerId,Survived
892,0
893,1
894,0
...
```

Key rules:
- Column names must match exactly what the competition specifies in the sample submission file
- Row count must equal the test set size — no more, no fewer
- No extra columns, no index column

## Code Competition Format

Some competitions require you to submit a notebook that produces predictions at runtime. In Code Competitions:
- Data is read from `/kaggle/input/<competition-slug>/`
- Output is written to `/kaggle/working/submission.csv`
- Internet access is disabled during execution
- All dependencies must be bundled or pre-installed in the Kaggle environment

## End-to-End Submission Workflow

1. Train your model on the training data
2. Generate predictions on the test set
3. Format predictions as a DataFrame with the required columns
4. Save to CSV: `df.to_csv("submission.csv", index=False)`
5. Submit via CLI: `kaggle competitions submit -c <name> -f submission.csv -m "description"`
6. Check your score: `kaggle competitions submissions -c <name>`

The toolkit's `SubmissionGenerator` handles steps 3–5 automatically.

## Managing the Daily Submission Limit

Most competitions allow 5–10 submissions per day. Tips:
- Use cross-validation locally to estimate performance before spending a submission
- Reserve submissions for meaningful experiments, not incremental tweaks
- Track every submission with a descriptive message so you can correlate scores to changes
- Late in a competition, save submissions for your best models rather than exploratory ones
