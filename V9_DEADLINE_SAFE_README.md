# V9 deadline-safe inference — immediate runbook

This is a **new inference-only** program. It does not retrain V8, rebuild indexes, contact any business database, or guarantee a leaderboard improvement. It reuses the V8 test SQLite database and the trained V8 XGBoost configuration, accelerates blocking with batched indexed exact-name/exact-address SQL queries, and scores a bounded number of candidates. Optional ANN is OFF by default. When its time budget expires, it writes V1 predictions for all remaining S1 entities.

**Required files:** place `model_v9_deadline_safe.py` alongside `model_v4_hybrid.py`, `model_v5_experimental.py`, `model_v6_fast.py`, `model_v7_fresh.py` in the project root. Existing folders must remain in place: `student_resource/artifacts_v8_test/` (test database and `s1_by_country/`), `student_resource/models_v8/`, `v1_backup/matching_results.tsv`, ideally `v1_backup/candidate_pairs.tsv`.

## Step 0 — stop old V8 but preserve ALL artifacts

In the terminal running the old test pipeline, press **Ctrl+C**, or stop just that Python process in Task Manager. Never delete the completed V8 database, indexes, encoders, training metadata or V1 backup. The new script does not depend on old V8's partial France prediction file.

## Step 1 — 2,000-row benchmark (~first 2,000 India queries)

Run from your project root in Windows CMD:

```cmd
python -u model_v9_deadline_safe.py --benchmark-rows 2000 --batch 400 --threads 8 --output student_resource\output_v9_benchmark > v9_benchmark.log 2>&1
powershell -Command "Get-Content v9_benchmark.log -Tail 20"
```

Read `BATCH` timing lines: `exact`, `ann`, `model`, total and **rows/s**. A 2,000-row sample doesn't predict performance for every country; use it to ensure there is no catastrophic bottleneck. The benchmark does NOT create a finished submission. On the full 1,732,544 S1 entities, 100/s means ~4.8 hours of active inference; 150/s means ~3.2 hours. If speed is slower, the budgeted process still produces complete output by falling back to V1.

## Step 2 — real run, exactly once in its own output directory

```cmd
python -u model_v9_deadline_safe.py --minutes 210 --batch 400 --threads 8 --output student_resource\output_v9 > v9.log 2>&1
```

Do not add `--ann` unless a separate ANN benchmark shows it is *faster enough* and useful. Exact-only is much more predictable and avoids loading millions of FAISS vectors into RAM. The default country priority is India, US, then France, to prioritize known training countries before spending the budget; output file row order can differ from original test input.

A complete copy of V1's two output files appears in `student_resource/output_v9/` **before inference** as a safe fallback. Do **not** assume those initial files contain V9 enhancements. Wait for `SUBMISSION READY` in `v9.log` to know final V9 outputs are finished. The script commits batch checkpoints with byte offsets, so if it stops unexpectedly, rerun the **same command** with unchanged settings and it will resume incomplete countries without duplicating rows.

The 210-minute budget limits active V9 inference only; allow time for startup, final file writing, validation, submission upload, and the required final code/methodology package. Shorten `--minutes` if there is less time remaining.

## Step 3 — official validator and submission

```cmd
python student_resource\utils\validate_submission.py --matching student_resource\output_v9\matching_results.tsv --candidate student_resource\output_v9\candidate_pairs.tsv --test-dir student_resource\dataset\test
```

If the supplied validator is in another location, adapt only the script path. Optional memory-intensive ID validation: append `--check-ids` if time and RAM permit. Ensure `PASS` before uploading.

For the **leaderboard portal**, upload `student_resource/output_v9/matching_results.tsv`. The final team ZIP (a separate deliverable in the provided challenge README) requires an `output/` folder containing **both** TSV files and also `code/business_entity_resolution/` with runnable source, README and requirements, plus the filled methodology template. A ZIP containing only the two TSV files is **not** the full required final package.

## AWS SageMaker decision

Run locally unless the same completed test SQLite database, country partitions, V8 model files and V1 fallback are *already* on a SageMaker instance with sufficient attached disk. Migrating and rebuilding millions of targets can waste the remaining time. The default pipeline is mostly SQLite and CPU/Python feature extraction; renting a GPU alone is unlikely to solve its bottleneck. A larger cloud CPU instance only pays off if you benchmark a faster per-batch rate after files are present. **Don't run V8 and V9 concurrently on 32 GB RAM.**

## Limitations

V9's exact-blocking recall is lower than full V8's multichannel retrieval, especially for misspelled names and addresses where *neither* field exactly matches. V8's validation F0.5 (approximately 0.94 on earlier train-based split) **does not apply directly to V9** and does not predict test leaderboard performance. Strong matching requires both normalized name and address to agree (or fuzzy near-agreement with no contradictory address numbers); other shortlisted exact-block candidates go through the saved V8 classifier. V1 predictions are kept when V9 has no confident replacement.
