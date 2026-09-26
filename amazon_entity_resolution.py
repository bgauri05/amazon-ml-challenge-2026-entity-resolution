#!/usr/bin/env python3
"""Amazon ML Challenge 2026: reproducible, disk-backed entity resolution.

Install: pip install duckdb lightgbm rapidfuzz numpy
Run: python amazon_entity_resolution.py --data-dir dataset --output-dir output \
         --work-dir work --threads 16 --train-rows 120000 --valid-rows 30000

The script creates BOTH required TSV files. It uses only supplied competition data.
No score is guaranteed: inspect retrieval recall and local macro F0.5 before submitting.
"""
from __future__ import annotations
import argparse
import csv
import logging
import os
from pathlib import Path
import time

import duckdb
import lightgbm as lgb
import numpy as np
from rapidfuzz import fuzz

LOG = logging.getLogger("entity_resolution")
FEATURES = [
    "name_wr", "name_ratio", "name_token", "name_partial", "name_core_wr",
    "address_wr", "address_token", "address_partial", "country_eq",
    "name_eq", "core_eq", "addr_eq", "number_eq", "first_token_eq",
    "name_len_ratio", "addr_len_ratio", "name_token_count_diff",
    "addr_token_count_diff", "candidate_routes", "s2", "s3",
]


def args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("dataset"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--work-dir", type=Path, default=Path("work"))
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--memory-limit", default="12GB", help="DuckDB memory limit; leave RAM for Python")
    p.add_argument("--train-rows", type=int, default=120000, help="S1 train sample size; 0 means all")
    p.add_argument("--valid-rows", type=int, default=30000, help="disjoint S1 validation size")
    p.add_argument("--batch", type=int, default=25000, help="feature/scoring batch size")
    p.add_argument("--partitions", type=int, default=24, help="test S1 hash partitions")
    p.add_argument("--max-bucket", type=int, default=150, help="ignore extremely broad blocking buckets")
    p.add_argument("--max-candidates", type=int, default=50, help="max candidates per S1")
    p.add_argument("--threshold", type=float, default=None, help="override validated threshold")
    p.add_argument("--skip-test", action="store_true", help="train and validate only")
    return p.parse_args()


def qpath(p: Path) -> str:
    return "'" + str(p.resolve()).replace("'", "''") + "'"


def normalize(col: str) -> str:
    # SQL expressions run in DuckDB's vectorized engine. Preserve country as free text.
    # translate folds common French accents; other scripts remain searchable by address.
    folded = f"translate(lower(coalesce({col}, '')), 'àâäáãåæçèéêëìíîïñòóôöõùúûüýÿœ', 'aaaaaaaceeeeiiiinooooouuuuyyo')"
    return f"trim(regexp_replace(regexp_replace({folded}, '[^a-z0-9\\p{{L}}\\p{{N}}]+', ' ', 'g'), ' +', ' ', 'g'))"


def create_data(db, root: Path, split: str):
    for source in (1, 2, 3):
        path = root / split / f"{split}_source{source}.tsv"
        if not path.is_file():
            raise FileNotFoundError(path)
        table = f"{split}_s{source}"
        db.execute(f"DROP TABLE IF EXISTS {table}")
        # Explicit tab delimiter and all-varchar columns avoid ID reinterpretation.
        db.execute(f"""CREATE TABLE {table} AS SELECT
            entity_id AS id, {normalize('business_name')} AS n,
            {normalize('business_address')} AS a,
            lower(trim(coalesce(country, ''))) AS c
            FROM read_csv({qpath(path)}, delim='\\t', header=true,
              all_varchar=true, null_padding=true, ignore_errors=false)""")
        db.execute(f"""CREATE OR REPLACE TABLE {table} AS SELECT *,
            trim(regexp_replace(n, '\\b(inc|incorporated|llc|llp|ltd|limited|private|pvt|corp|corporation|company|co|sa|sarl|sas)\\b', '', 'g')) AS core,
            regexp_extract(a, '\\b[0-9]+\\b', 0) AS num,
            split_part(n, ' ', 1) AS first,
            length(n) AS nl, length(a) AS al
            FROM {table}""")
        LOG.info("%s: %s rows", table, db.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
    db.execute(f"CREATE OR REPLACE TABLE {split}_other AS SELECT *, 2 AS src FROM {split}_s2 UNION ALL SELECT *, 3 AS src FROM {split}_s3")


def ground_truth(db, root: Path):
    path = root / "train" / "train_ground_truth.tsv"
    if not path.is_file():
        raise FileNotFoundError(path)
    db.execute(f"""CREATE OR REPLACE TABLE gt_raw AS SELECT source1_entity_id AS id,
      coalesce(matched_entity_ids, '') AS matches FROM read_csv({qpath(path)},
      delim='\\t', header=true, all_varchar=true, null_padding=true)""")
    db.execute("""CREATE OR REPLACE TABLE gt AS SELECT id AS s1, trim(t.mid) AS s23
      FROM gt_raw, unnest(string_split(matches, ',')) AS t(mid)
      WHERE length(trim(t.mid)) > 0""")


# Complementary, bounded blocking joins. No country name is hard-coded.
ROUTES = [
    ("exact_name", "n", "n", "length(l.n) >= 3"),
    ("core_name", "core", "core", "length(l.core) >= 3"),
    ("prefix8_num", "left(n, 8) || ':' || num", "left(n, 8) || ':' || num", "length(l.n) >= 8 AND length(l.num) > 0"),
    ("prefix5_num", "left(n, 5) || ':' || num", "left(n, 5) || ':' || num", "length(l.n) >= 5 AND length(l.num) > 0"),
    ("address_name", "a || ':' || left(n, 3)", "a || ':' || left(n, 3)", "length(l.a) >= 8 AND length(l.n) >= 3"),
    ("first_num", "first || ':' || num", "first || ':' || num", "length(l.first) >= 4 AND length(l.num) > 0"),
    ("name_address", "left(n, 9) || ':' || left(a, 8)", "left(n, 9) || ':' || left(a, 8)", "length(l.n) >= 9 AND length(l.a) >= 8"),
    ("suffix_num", "right(n, 8) || ':' || num", "right(n, 8) || ':' || num", "length(l.n) >= 8 AND length(l.num) > 0"),
    ("rare_address", "a", "a", "length(l.a) >= 14"),
]


def candidates(db, left: str, tag: str, max_bucket: int, max_candidates: int):
    """Materialize only the final candidate set fed to the classifier."""
    db.execute(f"DROP TABLE IF EXISTS {tag}_raw")
    db.execute(f"CREATE TABLE {tag}_raw(s1 VARCHAR, s23 VARCHAR, route INTEGER)")
    for i, (label, lk, rk, cond) in enumerate(ROUTES):
        LOG.info("%s: blocking route %s", tag, label)
        # Cap the RHS key frequency before joining; broad names cannot explode.
        db.execute(f"""INSERT INTO {tag}_raw
            WITH r AS (SELECT id, c, ({rk}) AS k FROM {tag}_r),
            limited AS (SELECT c, k FROM r GROUP BY c,k
                       HAVING count(*) BETWEEN 1 AND {max_bucket})
            SELECT l.id, r.id, {i} FROM {left} l
            JOIN limited b ON l.c=b.c AND ({lk})=b.k
            JOIN r ON r.c=b.c AND r.k=b.k
            WHERE {cond} AND length(b.k) >= 3""")
    # Route count rewards agreement of independent blocking methods.
    db.execute(f"""CREATE OR REPLACE TABLE {tag}_pairs AS
        WITH hits AS (SELECT s1,s23,count(DISTINCT route) AS routes
                     FROM {tag}_raw GROUP BY s1,s23),
        ranked AS (SELECT *, row_number() OVER(PARTITION BY s1 ORDER BY routes DESC, s23) AS rank
                   FROM hits)
        SELECT s1,s23,routes FROM ranked WHERE rank <= {max_candidates}""")
    db.execute(f"DROP TABLE {tag}_raw")
    LOG.info("%s: %s candidate pairs", tag, db.execute(f"SELECT count(*) FROM {tag}_pairs").fetchone()[0])


def setup_right(db, right: str, tag: str):
    db.execute(f"CREATE OR REPLACE VIEW {tag}_r AS SELECT * FROM {right}")


def pair_query(left: str, tag: str, extra: str = "") -> str:
    return f"""SELECT p.s1,p.s23,p.routes,
         l.n,l.core,l.a,l.c,l.num,l.first,l.nl,l.al,
         r.n,r.core,r.a,r.c,r.num,r.first,r.nl,r.al,r.src
      FROM {tag}_pairs p JOIN {left} l ON p.s1=l.id
      JOIN {tag}_r r ON p.s23=r.id {extra} ORDER BY p.s1,p.s23"""


def feature_rows(rows):
    x = np.empty((len(rows), len(FEATURES)), dtype=np.float32)
    for k, row in enumerate(rows):
        _, _, routes, n1, core1, a1, c1, num1, first1, nl1, al1, n2, core2, a2, c2, num2, first2, nl2, al2, src = row
        x[k] = (
            fuzz.WRatio(n1, n2), fuzz.ratio(n1, n2), fuzz.token_sort_ratio(n1, n2),
            fuzz.partial_ratio(n1, n2), fuzz.WRatio(core1, core2),
            fuzz.WRatio(a1, a2) if a1 and a2 else 0,
            fuzz.token_sort_ratio(a1, a2) if a1 and a2 else 0,
            fuzz.partial_ratio(a1, a2) if a1 and a2 else 0,
            c1 == c2, bool(n1) and n1 == n2, bool(core1) and core1 == core2,
            bool(a1) and a1 == a2, bool(num1) and num1 == num2,
            bool(first1) and first1 == first2,
            min(nl1,nl2)/max(1,nl1,nl2), min(al1,al2)/max(1,al1,al2),
            abs(len(n1.split())-len(n2.split())), abs(len(a1.split())-len(a2.split())),
            routes, src == 2, src == 3,
        )
    return x


def load_scored(db, query, batch, model=None, positives=None):
    cur = db.execute(query)
    ids, labels, arrays, scores = [], [], [], []
    while rows := cur.fetchmany(batch):
        xx = feature_rows(rows)
        ids.extend((r[0], r[1]) for r in rows)
        if positives is not None:
            labels.extend(int((r[0],r[1]) in positives) for r in rows)
            arrays.append(xx)
        if model is not None:
            scores.extend(model.predict(xx, num_threads=1))
    return ids, np.concatenate(arrays) if arrays else np.empty((0,len(FEATURES)),np.float32), np.asarray(labels,np.uint8), np.asarray(scores,np.float32)


def sample_training(db, n_train, n_valid):
    n = db.execute("SELECT count(*) FROM train_s1").fetchone()[0]
    if n_train == 0:
        n_train = max(0, n - n_valid)
    if n_train + n_valid > n:
        raise ValueError("Requested more train/validation S1 rows than available")
    # Stable hash order; validation entities are never used for fitting.
    db.execute(f"""CREATE OR REPLACE TABLE sampled AS
        SELECT *, row_number() OVER(ORDER BY hash(id)) AS sample_row
        FROM train_s1 QUALIFY sample_row <= {n_train+n_valid}""")
    db.execute(f"CREATE OR REPLACE TABLE fit_s1 AS SELECT * EXCLUDE(sample_row) FROM sampled WHERE sample_row <= {n_train}")
    db.execute(f"CREATE OR REPLACE TABLE val_s1 AS SELECT * EXCLUDE(sample_row) FROM sampled WHERE sample_row > {n_train}")
    return n_train, n_valid


def macro_score(all_ids, true_map, pred):
    s = 0.0
    for id in all_ids:
        actual = true_map.get(id, set())
        guessed = pred.get(id, set())
        if not actual and not guessed:
            s += 1.0
        elif actual and guessed:
            tp = len(actual & guessed)
            s += 1.25 * tp / (len(guessed) + 0.25 * len(actual))
    return s / max(1,len(all_ids))


def validate(db, ids, scores, train_tag):
    truth = {}
    for s1, s23 in db.execute("SELECT g.s1,g.s23 FROM gt g JOIN val_s1 l ON l.id=g.s1").fetchall():
        truth.setdefault(s1,set()).add(s23)
    all_ids = [r[0] for r in db.execute("SELECT id FROM val_s1").fetchall()]
    candidate_set = set(ids)
    covered = sum((s1,s23) in candidate_set for s1, links in truth.items() for s23 in links)
    total_links = sum(map(len, truth.values()))
    LOG.info("Validation candidate recall: %.5f (%d/%d)", covered/max(1,total_links),covered,total_links)
    best = (-1,None)
    for threshold in np.arange(.40,.996,.015):
        pred = {}
        for (s1,s23),score in zip(ids,scores):
            if score >= threshold:
                pred.setdefault(s1,set()).add(s23)
        result = macro_score(all_ids,truth,pred)
        if result > best[0]: best = (result,float(threshold))
    LOG.info("Validation macro F0.5: %.6f at threshold %.3f", *best)
    return best[1]


def train(db, a):
    n_fit,n_val = sample_training(db,a.train_rows,a.valid_rows)
    if n_fit == 0 or n_val == 0:
        raise ValueError("Both fit and validation samples must have rows")
    setup_right(db,"train_other","fit")
    candidates(db,"fit_s1","fit",a.max_bucket,a.max_candidates)
    # Inject labeled positives for FIT ONLY. Validation/inference never receive oracle pairs.
    db.execute("""CREATE OR REPLACE TABLE fit_pairs AS
      SELECT s1,s23,max(routes) AS routes FROM (
        SELECT * FROM fit_pairs
        UNION ALL SELECT g.s1,g.s23,1 FROM gt g JOIN fit_s1 f ON f.id=g.s1
      ) GROUP BY s1,s23""")
    positive = set(db.execute("SELECT g.s1,g.s23 FROM gt g JOIN fit_s1 l ON l.id=g.s1").fetchall())
    ids, x, y, _ = load_scored(db,pair_query("fit_s1","fit"),a.batch,positives=positive)
    LOG.info("Fit pairs: %d, positives: %d",len(y),int(y.sum()))
    if y.sum() < 2 or y.sum() == len(y):
        raise ValueError("Insufficient positive/negative training pairs. Increase --train-rows or blocking coverage")
    del ids, positive
    model = lgb.LGBMClassifier(n_estimators=550, learning_rate=.06, num_leaves=63,
        max_depth=-1, min_child_samples=80, colsample_bytree=.85, reg_lambda=8,
        n_jobs=a.threads, verbosity=-1)
    model.fit(x,y,feature_name=FEATURES)
    del x,y
    model.booster_.save_model(str(a.work_dir / "lightgbm.txt"))
    db.execute("DROP TABLE fit_pairs")
    setup_right(db,"train_other","val")
    candidates(db,"val_s1","val",a.max_bucket,a.max_candidates)
    ids,_,_,scores = load_scored(db,pair_query("val_s1","val"),a.batch,model=model)
    threshold = validate(db,ids,scores,"val")
    del ids,scores
    db.execute("DROP TABLE val_pairs")
    return model, threshold


def write_test(db, model, a, threshold):
    output = a.output_dir
    output.mkdir(parents=True,exist_ok=True)
    match_path = output / "matching_results.tsv"
    cand_path = output / "candidate_pairs.tsv"
    # Temporary output ensures interrupted runs do not look like finished submissions.
    mtmp, ctmp = output / "matching_results.tsv.partial", output / "candidate_pairs.tsv.partial"
    score_path = a.work_dir / "test_scores.tsv"
    with ctmp.open("w",newline="",encoding="utf-8") as cf, score_path.open("w",newline="",encoding="utf-8") as sf:
        cw=csv.writer(cf,delimiter="\t",lineterminator="\n")
        sw=csv.writer(sf,delimiter="\t",lineterminator="\n")
        cw.writerow(("source1_entity_id","candidate_entity_ids"))
        sw.writerow(("s1","s23","score"))
        for part in range(a.partitions):
            t0=time.time()
            db.execute(f"CREATE OR REPLACE TABLE part_s1 AS SELECT * FROM test_s1 WHERE hash(id)%{a.partitions}={part}")
            setup_right(db,"test_other","part")
            candidates(db,"part_s1","part",a.max_bucket,a.max_candidates)
            query = pair_query("part_s1","part")
            cur = db.execute(query)
            # Candidate rows are sorted by S1 and capped at max_candidates; accumulate one part.
            matches,candidates_by_s1={},{}
            while rows := cur.fetchmany(a.batch):
                xx=feature_rows(rows)
                predictions=model.predict(xx,num_threads=1)
                for r, score in zip(rows,predictions):
                    s1,s23=r[:2]
                    candidates_by_s1.setdefault(s1,[]).append(s23)
                    if score >= threshold: matches.setdefault(s1,[]).append((s23,float(score)))
            for s1,items in matches.items():
                sw.writerows((s1,s23,score) for s23,score in items)
            for (s1,) in db.execute("SELECT id FROM part_s1 ORDER BY id").fetchall():
                cw.writerow((s1, ",".join(sorted(set(candidates_by_s1.get(s1,[]))))))
            LOG.info("Test partition %d/%d: %d references, %d candidates, %.1fs",
                part+1,a.partitions,db.execute("SELECT count(*) FROM part_s1").fetchone()[0],
                sum(map(len,candidates_by_s1.values())),time.time()-t0)
            db.execute("DROP TABLE part_pairs")
    db.execute(f"""CREATE OR REPLACE TABLE test_predictions AS SELECT * FROM
      read_csv({qpath(score_path)}, delim='\\t',header=true,
      columns={{'s1':'VARCHAR','s23':'VARCHAR','score':'FLOAT'}})""")
    db.execute("""CREATE OR REPLACE TABLE final_predictions AS
      SELECT s1,s23 FROM (SELECT *,row_number() OVER(PARTITION BY s23 ORDER BY score DESC,s1) AS rn
      FROM test_predictions) WHERE rn=1""")
    with mtmp.open("w",newline="",encoding="utf-8") as mf:
        mw=csv.writer(mf,delimiter="\t",lineterminator="\n")
        mw.writerow(("source1_entity_id","matched_entity_ids"))
        cur=db.execute("""SELECT l.id,coalesce(string_agg(p.s23, ',' ORDER BY p.s23),'')
            FROM test_s1 l LEFT JOIN final_predictions p ON l.id=p.s1
            GROUP BY l.id ORDER BY l.id""")
        while rows := cur.fetchmany(50000):
            mw.writerows(rows)
    os.replace(mtmp,match_path)
    os.replace(ctmp,cand_path)
    LOG.info("Wrote %s and %s",match_path,cand_path)


def main():
    a=args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    a.work_dir.mkdir(parents=True,exist_ok=True)
    db=duckdb.connect(str(a.work_dir / "entity_resolution.duckdb"))
    db.execute(f"SET threads={max(1,a.threads)}")
    db.execute("SET memory_limit=?", [a.memory_limit])
    db.execute(f"SET temp_directory={qpath(a.work_dir / 'spill')}")
    create_data(db,a.data_dir,"train")
    ground_truth(db,a.data_dir)
    model,chosen=train(db,a)
    threshold = a.threshold if a.threshold is not None else chosen
    LOG.info("Using score threshold %.4f",threshold)
    if not a.skip_test:
        create_data(db,a.data_dir,"test")
        write_test(db,model,a,threshold)
    db.close()


if __name__ == "__main__":
    main()
