# Pack the official sample agent into submissions/submission.tar.gz (what submission 58332051 used).
# Run from competitions/pokemon-ptcg-playground. Replace main.py / deck.csv to ship a new agent.
$s = "data\sample_submission\sample_submission\sample_submission"
New-Item -ItemType Directory -Force submissions | Out-Null
tar -czf submissions\submission.tar.gz -C $s main.py deck.csv cg
tar -tzf submissions\submission.tar.gz
