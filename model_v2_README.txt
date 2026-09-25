Amazon ML Challenge entity resolution: model_v2.py

WHAT CHANGED
- Hybrid retrieval: CPU FAISS HNSW plus SQLite exact canonical-name and canonical-address matches.
- Better hard-negative sampling: ground-truth positives not retrieved are NOT injected into classifier training.
- Extra canonical-key and containment features.
- Encoder and completed training/test FAISS indexes saved for reuse.
- Country-agnostic indexing; test France is indexed automatically.
- Higher efSearch by default; tune top-k and inspect recall/F0.5.

INSTALL (use the same Python environment that ran your previous model):
python -m pip install numpy pandas scipy scikit-learn faiss-cpu rapidfuzz joblib xgboost torch

FIRST RUN, VALIDATION ONLY:
python model_v2.py --sample 10000 --top-k 30 --skip-test

IF VALIDATION IS IMPROVED, FULL SUBMISSION:
python model_v2.py --sample 30000 --top-k 30

REUSE CHECKPOINTS (same --dim and --svd-samples):
Simply run again without --rebuild. Cached FAISS indexes and encoder are loaded.

REBUILD FROM SCRATCH:
python model_v2.py --sample 10000 --top-k 30 --skip-test --rebuild

FILES:
The script uses dataset/ or student_resource/dataset/ under the directory where model_v2.py resides.
It writes artifacts_gpu/ and output/ beside the dataset folder.
The complete training and test target indexes are separate. First run still rebuilds both datasets.

IMPORTANT:
- This is an experimental revision, not a proven 0.98+ solution.
- Runtime, RAM and disk usage can be substantial for 10M target records.
- Cache identity depends on --dim and --svd-samples. After code/data changes, use --rebuild.
- For a new test dataset, delete the old test cache or use --rebuild.
- Exact blocking is capped for common keys; benchmark recall and runtime.
- FAISS runs on CPU on native Windows. PyTorch embedding projection and XGBoost use CUDA.
- Output: output/matching_results.tsv and output/candidate_pairs.tsv.
- Run the official validator before submission.
