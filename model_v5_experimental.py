#!/usr/bin/env python3
r"""Model V5: retrieval expansion + hard-negative mining over the user's cached V4 indexes.

EXPERIMENTAL: no accuracy is guaranteed. Never use validation truth for retrieval
or training. Requires model_v4_hybrid.py and its previously built V4 caches in
student_resource/artifacts_v3 (V4's actual directory name).
"""
from __future__ import annotations
import argparse
import csv
import gc
import logging
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import sqlite3
from concurrent.futures import ThreadPoolExecutor
import faiss
import joblib
import numpy as np
import torch
import xgboost as xgb
from rapidfuzz import fuzz, distance
from sklearn.model_selection import train_test_split

import model_v4_hybrid as v4
import __main__  # V4 was launched as a script, so its joblib pickle refers to __main__.Encoder.
__main__.Encoder = v4.Encoder

LOG = logging.getLogger('model_v5')
SEED = 42
BASE_FEATURES = v4.features
BASE_STORE = v4.TargetStore
NAME_STOP = {'the','company','co','private','limited','pvt','ltd','llc','inc',
             'enterprises','enterprise','services','service','center','centre',
             'business','consultants','consulting','com','www','group','corp'}
ADDR_STOP = {'the','near','road','street','avenue','floor','building','private',
             'company','india','nagar','district','city','state','unit','sector',
             'lane','west','east','north','south','maharashtra','rajasthan',
             'delhi','telangana','gujarat','karnataka','india','mumbai'}


def clean_alnum(s):
    return ''.join(c for c in s if c.isalnum())


def initials(s):
    return ''.join(t[0] for t in s.split() if t)


def token_overlap(a, b):
    ta, tb = set(a.split()), set(b.split())
    return len(ta & tb) / max(1, min(len(ta), len(tb))) if ta and tb else 0.0


def extra_features(a, b, similarity):
    """Independent, numeric-aware similarity signals. No ground-truth inputs."""
    f = BASE_FEATURES(a, b, similarity)
    na, nb = v4.name_key(a[1]), v4.name_key(b[1])
    aa, ab = v4.addr_key(a[2]), v4.addr_key(b[2])
    nta, ntb = na.split(), nb.split()
    ata, atb = aa.split(), ab.split()
    numa = re.findall(r'\d+', aa)
    numb = re.findall(r'\d+', ab)
    znuma = {t.lstrip('0') or '0' for t in numa}
    znumb = {t.lstrip('0') or '0' for t in numb}
    core_a = ' '.join(t for t in nta if t not in NAME_STOP)
    core_b = ' '.join(t for t in ntb if t not in NAME_STOP)
    addr_a = ' '.join(t for t in ata if t not in ADDR_STOP and not t.isdigit())
    addr_b = ' '.join(t for t in atb if t not in ADDR_STOP and not t.isdigit())
    # Separate raw similarity from normalized cross-field evidence.
    f.extend([
        fuzz.ratio(clean_alnum(na), clean_alnum(nb))/100,
        fuzz.ratio(clean_alnum(aa), clean_alnum(ab))/100,
        fuzz.WRatio(core_a, core_b)/100 if core_a and core_b else 0.,
        fuzz.token_set_ratio(core_a, core_b)/100 if core_a and core_b else 0.,
        fuzz.WRatio(addr_a, addr_b)/100 if addr_a and addr_b else 0.,
        token_overlap(core_a, core_b),
        token_overlap(addr_a, addr_b),
        float(bool(znuma and znumb and znuma == znumb)),
        float(bool(znuma and znumb and znuma.issubset(znumb))),
        float(bool(znuma and znumb and znumb.issubset(znuma))),
        float(bool(znuma and znumb and not znuma.intersection(znumb))),
        len(znuma & znumb) / max(1, len(znuma | znumb)),
        fuzz.ratio(initials(na), initials(nb))/100,
        distance.JaroWinkler.normalized_similarity(clean_alnum(na), clean_alnum(nb)),
        float(bool(core_a and core_b and clean_alnum(core_a) == clean_alnum(core_b))),
        float(bool(aa and ab and clean_alnum(aa) == clean_alnum(ab))),
        float(bool(na and nb and clean_alnum(na) in clean_alnum(nb))),
        float(bool(na and nb and clean_alnum(nb) in clean_alnum(na))),
        float(bool(numa and numb and (numa[0].lstrip('0') or '0') == (numb[0].lstrip('0') or '0'))),
    ])
    return f


class ExpandedStore(BASE_STORE):
    """Reuse cached V4 full-corpus indexes and add independent alternate FTS probes with multithreaded SQLite access."""
    extras_per_probe = 75
    extra_cap = 750
    extra_enabled = True

    def __init__(self, db_path, dim, cpu_threads=12, *args, **kwargs):
        super().__init__(db_path, dim, *args, **kwargs)
        self.db_path = Path(db_path)
        self.cpu_threads = cpu_threads

    def _get_conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA query_only=ON")
        return conn

    @staticmethod
    def probes(r):
        name = [t for t in v4.name_key(r[1]).split()
                if len(t) >= 3 and t not in NAME_STOP]
        addr = [t for t in v4.addr_key(r[2]).split()
                if len(t) >= 3 and t not in ADDR_STOP]
        nums = [s for s in re.findall(r'\d+', r[2]) if len(s) >= 2]
        p = []
        if len(name) >= 2:
            p.append(('name', name[0], name[-1]))
            p.append(('name', name[-2], name[-1]))
        if len(name) >= 3:
            p.append(('name', name[1], name[2]))
        if len(addr) >= 2:
            p.append(('addr', addr[0], addr[-1]))
            p.append(('addr', addr[-2], addr[-1]))
        if nums and addr:
            p.append(('addr', nums[-1], addr[-1]))
        if len(addr) >= 3:
            p.append(('addr', addr[1], addr[2]))
        seen = set()
        for field, t1, t2 in p:
            if t1 == t2:
                continue
            key = (field, tuple(sorted((t1, t2))))
            if key not in seen:
                seen.add(key)
                yield key

    def fts_candidates_parallel(self, records, per_probe=55):
        """Multithreaded FTS primary probes selecting pre-computed keys from SQLite."""
        sql = ("SELECT t.id,t.name,t.addr,t.name_key,t.addr_key FROM targets_fts "
               "JOIN targets t ON t.rowid=targets_fts.rowid "
               "WHERE targets_fts MATCH ? AND t.country=? "
               "ORDER BY bm25(targets_fts) LIMIT ?")

        def process_rec(r, conn):
            name_tokens = sorted({t for t in re.findall(r'[a-z0-9]+', v4.name_key(r[1]))
                                  if len(t) >= 4 and t not in ADDR_STOP | NAME_STOP},
                                 key=lambda x: (-len(x), x))[:4]
            addr_tokens = sorted({t for t in re.findall(r'[a-z0-9]+', v4.addr_key(r[2]))
                                  if len(t) >= 4 and t not in ADDR_STOP | NAME_STOP},
                                 key=lambda x: (-len(x), x))[:4]
            numbers = [x for x in re.findall(r'\d+', r[2]) if len(x) >= 2]
            probes = []
            for field, tokens in (('name', name_tokens), ('addr', addr_tokens)):
                if len(tokens) >= 2: probes.append((field, tokens[:2]))
                if tokens and len(tokens[0]) >= 5: probes.append((field, tokens[:1]))
                if field == 'addr' and numbers and tokens: probes.append((field, [numbers[0], tokens[0]]))
            by_field = {'name': [], 'addr': []}
            rec_ak = v4.addr_key(r[2])
            rec_nk = v4.name_key(r[1])
            for field, tokens in probes[:5]:
                expr = ' AND '.join(f'{field}:"{t}"' for t in tokens)
                found = conn.execute(sql, (expr, r[3], per_probe * 3)).fetchall()
                if len(found) > per_probe:
                    if field == 'name':
                        found = sorted(found, key=lambda x: fuzz.WRatio(rec_ak, x[4]), reverse=True)[:per_probe]
                    else:
                        found = sorted(found, key=lambda x: fuzz.WRatio(rec_nk, x[3]), reverse=True)[:per_probe]
                by_field[field].extend((cid, -1.0) for cid, _, _, _, _ in found)
            rec_res = []
            n = max(len(by_field['name']), len(by_field['addr']))
            for j in range(n):
                for field in ('name', 'addr'):
                    if j < len(by_field[field]):
                        rec_res.append(by_field[field][j])
            return rec_res

        def worker_chunk(chunk):
            conn = self._get_conn()
            res = [process_rec(r, conn) for r in chunk]
            conn.close()
            return res

        workers = getattr(self, 'cpu_threads', 12)
        chunk_size = max(1, len(records) // workers)
        chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for chunk_res in executor.map(worker_chunk, chunks):
                results.extend(chunk_res)
        return results

    def alternate_fts_parallel(self, records):
        """Multithreaded alternate FTS probes selecting pre-computed keys from SQLite."""
        stmt = ('SELECT t.id,t.name,t.addr,t.name_key,t.addr_key FROM targets_fts '
                'JOIN targets t ON t.rowid=targets_fts.rowid '
                'WHERE targets_fts MATCH ? AND t.country=? '
                'ORDER BY bm25(targets_fts) LIMIT ?')

        def process_rec(r, conn):
            name_hits, addr_hits = [], []
            probes = list(self.probes(r))[:7]
            rec_ak = v4.addr_key(r[2])
            rec_nk = v4.name_key(r[1])
            for field, tokens in probes:
                q = ' AND '.join(f'{field}:"{t}"' for t in tokens)
                hits = conn.execute(stmt, (q, r[3], self.extras_per_probe * 2)).fetchall()
                if hits:
                    if field == 'name':
                        hits = sorted(hits, key=lambda x: (fuzz.WRatio(rec_ak, x[4]),
                                                         fuzz.WRatio(rec_nk, x[3])),
                                      reverse=True)[:self.extras_per_probe]
                        name_hits.extend(x[0] for x in hits)
                    else:
                        hits = sorted(hits, key=lambda x: (fuzz.WRatio(rec_nk, x[3]),
                                                         fuzz.WRatio(rec_ak, x[4])),
                                      reverse=True)[:self.extras_per_probe]
                        addr_hits.extend(x[0] for x in hits)
            rec_res = []
            for j in range(max(len(name_hits), len(addr_hits))):
                if j < len(name_hits):
                    rec_res.append(name_hits[j])
                if j < len(addr_hits):
                    rec_res.append(addr_hits[j])
            return rec_res

        def worker_chunk(chunk):
            conn = self._get_conn()
            res = [process_rec(r, conn) for r in chunk]
            conn.close()
            return res

        workers = getattr(self, 'cpu_threads', 12)
        chunk_size = max(1, len(records) // workers)
        chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for chunk_res in executor.map(worker_chunk, chunks):
                results.extend(chunk_res)
        return results

    def retrieve(self, records, encoder, top_k):
        output = [[] for _ in records]
        vectors = encoder.transform(records)
        grouped = defaultdict(list)
        for i, r in enumerate(records):
            grouped[r[3]].append(i)
        for country, positions in grouped.items():
            q = np.ascontiguousarray(vectors[positions], dtype=np.float32)
            for source in (2, 3):
                key = (source, country)
                if key not in self.indexes or self.indexes[key].ntotal == 0:
                    continue
                count = min(top_k, self.indexes[key].ntotal)
                scores, labels = self.indexes[key].search(q, count)
                ids = self.ids[key]
                for local_i, original_i in enumerate(positions):
                    matches = []
                    for label, score in zip(labels[local_i], scores[local_i]):
                        if label < 0:
                            continue
                        matches.append((ids[int(label)], float(score)))
                    output[original_i].extend(matches)

        exact = self.exact_candidates(records, cap=self.exact_cap)
        lexical = self.fts_candidates_parallel(records, per_probe=self.fts_per_probe)

        for i, extra in enumerate(exact):
            extra.extend(lexical[i])
            existing = {cid for cid, _ in output[i]}
            for cid, _ in extra:
                if cid not in existing:
                    output[i].append((cid, -1.0))
                    existing.add(cid)
            if self.max_candidates > 0:
                output[i] = output[i][:max(self.max_candidates, 2 * top_k)]

        if self.extra_enabled:
            new_fts = self.alternate_fts_parallel(records)
            for i, row in enumerate(output):
                known = {cid for cid, _ in row}
                for cid in new_fts[i]:
                    if len(row) >= self.extra_cap:
                        break
                    if cid not in known:
                        row.append((cid, -1.0))
                        known.add(cid)
        return output


def format_progress(stage: str, current: int, total: int, start_time: float, extra: str = "") -> str:
    elapsed = time.time() - start_time
    rate = current / max(1e-5, elapsed)
    remaining = (total - current) / max(1e-5, rate) if current > 0 else 0
    pct = (current / total) * 100 if total > 0 else 100.0
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    eta_str = time.strftime("%H:%M:%S", time.gmtime(remaining))
    msg = f"[{time.strftime('%H:%M:%S')}] [{stage}] {current}/{total} ({pct:.1f}%) | Elapsed: {elapsed_str} | ETA: {eta_str} | {rate:.1f} rec/s"
    if extra:
        msg += f" | {extra}"
    return msg


def fit_xgb(X, y, device, n_estimators, threads, stage_name="Baseline XGBoost"):
    print(f"[{time.strftime('%H:%M:%S')}] [STAGE: {stage_name}] Starting GPU XGBoost training (tree_method='hist', device='cuda') on {len(y)} pairs (Pos: {int(y.sum())}, Neg: {int((y==0).sum())})...", flush=True)
    LOG.info("Starting %s GPU training (tree_method='hist', device='cuda')", stage_name)
    t0 = time.time()
    model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=8,
        learning_rate=.045, min_child_weight=2, reg_lambda=5, subsample=.85,
        colsample_bytree=.90, tree_method='hist', device='cuda',
        eval_metric='logloss', random_state=SEED, n_jobs=threads)
    model.fit(X, y, verbose=False)
    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] [STAGE: {stage_name}] Finished GPU XGBoost training in {elapsed:.2f}s", flush=True)
    LOG.info("Finished %s GPU training in %.2fs", stage_name, elapsed)
    model.set_params(device='cpu')
    return model


def evaluate(scored, truth, threshold):
    vals = []
    fp = fn = tp = 0
    for rid, pairs in scored.items():
        predicted = {cid for cid, prob in pairs if prob >= threshold}
        actual = truth.get(rid,set())
        vals.append(v4.entity_f05(actual,predicted))
        fp += len(predicted - actual)
        fn += len(actual - predicted)
        tp += len(actual & predicted)
    return float(np.mean(vals)), (tp,fp,fn)


def tune(scored, truth):
    # Source1-entity macro scoring, not global pairwise F0.5.
    thresholds = np.r_[np.arange(.60,.981,.02),np.arange(.80,.951,.005)]
    outcomes = [(evaluate(scored,truth,float(t))[0],float(t)) for t in np.unique(thresholds)]
    return max(outcomes, key=lambda x:x[0])


def mine_negatives(records, candidates, truth, store, model, per_entity=10, batch_size=100):
    """Mine *retrieved* false matches using training truth only, in bounded batches."""
    xx=[]
    yy=[]
    total=len(records)
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] [STAGE: Hard-Negative Mining] Starting mining across {total} S1 records...", flush=True)
    for start in range(0,total,batch_size):
        rr = records[start:start+batch_size]
        cc = candidates[start:start+batch_size]
        ss = v4.score_batch(rr,cc,store,model)
        pairs=[]
        for r in rr:
            gt = truth.get(r[0],set())
            ranked = sorted(((cid,p) for cid,p in ss[r[0]] if cid not in gt),
                            key=lambda x:-x[1])
            # Hardest high-scoring examples, plus several medium-scoring ones.
            chosen = ranked[:per_entity]
            if len(ranked)>per_entity+3:
                offset = max(per_entity, len(ranked)//3)
                chosen += ranked[offset:offset+3]
            pairs += [(r,cid) for cid,_ in chosen]
        targets=store.get_many([cid for _,cid in pairs])
        sim_by_id={r[0]:dict(c) for r,c in zip(rr,cc)}
        for r,cid in pairs:
            b=targets.get(cid)
            if b is not None:
                xx.append(extra_features(r,b,sim_by_id[r[0]][cid]))
                yy.append(0)
        curr = min(total, start + len(rr))
        msg = format_progress("Hard-Neg Mining", curr, total, t_start, f"Mined: {len(yy)} negs")
        print(msg, flush=True)
        LOG.info(msg)
    return np.asarray(xx,dtype=np.float32),np.asarray(yy,dtype=np.int8)


def diagnostic(scored, records, candidates, truth, store, threshold, output):
    """Separate misses caused by retrieval from classifier errors and write FP examples."""
    output.mkdir(parents=True,exist_ok=True)
    by_id={r[0]:r for r in records}
    retrieved={r[0]:{cid for cid,_ in row} for r,row in zip(records,candidates)}
    problems=[]
    counts=Counter()
    for rid,pairs in scored.items():
        expected=truth.get(rid,set())
        pred={cid for cid,p in pairs if p>=threshold}
        probabilities=dict(pairs)
        for cid in expected-pred:
            kind='retrieval_miss' if cid not in retrieved[rid] else 'classifier_false_negative'
            counts[kind]+=1
            if len(problems)<2500:
                problems.append((rid,cid,kind,probabilities.get(cid,-1.)))
        for cid in pred-expected:
            counts['classifier_false_positive']+=1
            if len(problems)<2500:
                problems.append((rid,cid,'classifier_false_positive',probabilities[cid]))
    details=store.get_many([cid for _,cid,_,_ in problems])
    with (output/'v5_error_audit.tsv').open('w',encoding='utf8',newline='') as f:
        writer=csv.writer(f,delimiter='\t',lineterminator='\n')
        writer.writerow(['s1_id','s1_name','s1_address','target_id','target_name',
                         'target_address','error','predicted_probability'])
        for rid,cid,kind,prob in problems:
            l=by_id[rid]; t=details.get(cid)
            writer.writerow([rid,l[1],l[2],cid,t[1] if t else '',
                             t[2] if t else '',kind,round(prob,5)])
    LOG.info('Validation error audit: %s', dict(counts))
    return counts


def main():
    parser=argparse.ArgumentParser(description='Experimental V5 using cached V4 indexes')
    parser.add_argument('--data',type=Path,default=Path('student_resource/student_resource/dataset'))
    parser.add_argument('--sample',type=int,default=0,
                        help='Sample size (0 or negative loads ALL available S1 training records)')
    parser.add_argument('--top-k',type=int,default=40)
    parser.add_argument('--dim',type=int,default=64)
    parser.add_argument('--svd-samples',type=int,default=60000)
    parser.add_argument('--cpu-threads',type=int,default=12)
    parser.add_argument('--query-batch',type=int,default=150)
    parser.add_argument('--max-candidates',type=int,default=520)
    parser.add_argument('--extra-cap',type=int,default=700)
    parser.add_argument('--fts-per-probe',type=int,default=55)
    parser.add_argument('--extra-probe-results',type=int,default=65)
    parser.add_argument('--no-extra-retrieval',action='store_true')
    parser.add_argument('--retrieval-only',action='store_true')
    parser.add_argument('--mining-per-entity',type=int,default=10)
    parser.add_argument('--skip-mining',action='store_true')
    parser.add_argument('--full-test',action='store_true')
    parser.add_argument('--candidate-cache',action='store_true',
                        help='Save/load candidate lists for this exact configuration/sample')
    args=parser.parse_args()
    if not (args.data / 'train' / 'train_ground_truth.tsv').exists():
        fallback = Path('student_resource/student_resource/dataset')
        if (fallback / 'train' / 'train_ground_truth.tsv').exists():
            args.data = fallback
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    faiss.omp_set_num_threads(args.cpu_threads)
    torch.set_num_threads(args.cpu_threads)
    # V4 actually wrote its caches to artifacts_v3, NOT artifacts_v4.
    work=args.data.parent/'artifacts_v3'
    out=args.data.parent/'output_v5'
    model_dir=args.data.parent/'models_v5'
    out.mkdir(parents=True,exist_ok=True)
    model_dir.mkdir(parents=True,exist_ok=True)
    ec=work/f'encoder_d{args.dim}_s{args.svd_samples}.joblib'
    ix=work/f'train_indexes_d{args.dim}_s{args.svd_samples}'/'complete.joblib'
    db=work/f'train_targets_d{args.dim}_s{args.svd_samples}.sqlite'
    if not(ec.exists() and ix.exists() and db.exists()):
        parser.error(f'Missing V4 caches in {work}. Finish one V4 run first using matching --dim/--svd-samples.')
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOG.info('Loading V4 cached encoder and 10M-row indexes; PyTorch device=%s',dev)
    encoder=joblib.load(ec)
    encoder.device=dev
    encoder.projection=torch.from_numpy(encoder.svd.components_.T.copy()).to(dev)
    ExpandedStore.extras_per_probe=args.extra_probe_results
    ExpandedStore.extra_cap=args.extra_cap
    ExpandedStore.extra_enabled=not args.no_extra_retrieval
    # Override module globals because V4's training/inference helpers refer to
    # their own module's features() and TargetStore symbols at call time.
    v4.features=extra_features
    v4.TargetStore=ExpandedStore
    store=ExpandedStore(db,args.dim,ef_search=192,cache_dir=work/f'train_indexes_d{args.dim}_s{args.svd_samples}',
         max_candidates=args.max_candidates,exact_cap=70,fts_per_probe=args.fts_per_probe)
    store.build(args.data,'train',encoder)
    if args.sample and args.sample > 0:
        print(f"[{time.strftime('%H:%M:%S')}] [Dataset] Sampling {args.sample} S1 records from disk...", flush=True)
        sampled=v4.sample_records(v4.source_path(args.data,'train',1),args.sample)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] [Dataset] Loading ALL available S1 training records from disk...", flush=True)
        sampled=list(v4.iter_records(v4.source_path(args.data,'train',1)))
    train,valid=train_test_split(sampled,test_size=.2,random_state=SEED)
    
    s1_count = len(sampled)
    s2_count = store.conn.execute("SELECT COUNT(*) FROM targets WHERE source=2").fetchone()[0]
    s3_count = store.conn.execute("SELECT COUNT(*) FROM targets WHERE source=3").fetchone()[0]
    
    print("\n================================================", flush=True)
    print("FULL DATASET RUN", flush=True)
    print("================================================", flush=True)
    print(f"S1 training records: {s1_count:,}", flush=True)
    print(f"S2 training records: {s2_count:,}", flush=True)
    print(f"S3 training records: {s3_count:,}", flush=True)
    print("\nCandidate retrieval: processing ALL S1 training records", flush=True)
    print("================================================\n", flush=True)
    
    LOG.info("Full Dataset Run | S1: %d, S2: %d, S3: %d", s1_count, s2_count, s3_count)
    truth=v4.read_truth(args.data/'train'/'train_ground_truth.tsv')
    key=f's{args.sample}_k{args.top_k}_m{args.max_candidates}_e{args.extra_cap}_p{args.extra_probe_results}_x{int(ExpandedStore.extra_enabled)}'
    def load_candidates(split_name,rr):
        p=out/f'{split_name}_candidates_{key}.joblib'
        if args.candidate_cache and p.exists():
            print(f"[{time.strftime('%H:%M:%S')}] [Candidate Retrieval] Loading {split_name} cached candidates from disk ({len(rr)} S1)", flush=True)
            LOG.info('Loading %s cached candidates',split_name)
            return joblib.load(p)
        print(f"[{time.strftime('%H:%M:%S')}] [Candidate Retrieval] Starting {split_name} retrieval for {len(rr)} S1 records...", flush=True)
        LOG.info('Retrieving %s candidates (%d S1)',split_name,len(rr))
        found=[]
        t_start = time.time()
        for start in range(0,len(rr),args.query_batch):
            stop = min(len(rr), start + args.query_batch)
            found += store.retrieve(rr[start:stop],encoder,args.top_k)
            msg = format_progress(f"Retrieval:{split_name}", stop, len(rr), t_start)
            print(msg, flush=True)
            LOG.info(msg)
        if args.candidate_cache:
            joblib.dump(found,p,compress=0)
        return found
    tr_candidates=load_candidates('train',train)
    v4.candidate_recall(train,tr_candidates,truth)
    vl_candidates=load_candidates('valid',valid)
    v4.candidate_recall(valid,vl_candidates,truth)
    ceiling=v4.retrieval_oracle_f05(valid,vl_candidates,truth)
    v4.retrieval_diagnostics(valid,vl_candidates,truth,store,out/'validation_retrieval_misses_v5.tsv',300)
    LOG.info('Oracle ceiling %.5f; requested 0.987 requires oracle at least 0.987',ceiling)
    if args.retrieval_only:
        LOG.info('Retrieval-only experiment finished; classifier not trained')
        store.close()
        return
    print(f"[{time.strftime('%H:%M:%S')}] [STAGE 2: Feature Matrix] Building feature matrices...", flush=True)
    X,y=v4.training_arrays(train,tr_candidates,truth,store,max_neg=24)
    LOG.info('Initial training pairs: %d (positives %d, negatives %d)',len(y),int(y.sum()),int((y==0).sum()))
    device='cuda'
    baseline=fit_xgb(X,y,device,650,args.cpu_threads,stage_name="Baseline XGBoost")
    # Split the V4 validation population: tuning and a distinct, untouched report subset.
    tune_refs,report_refs=train_test_split(valid,test_size=.50,random_state=SEED+1)
    tune_ids={r[0] for r in tune_refs}
    report_ids={r[0] for r in report_refs}
    def score_validation(model, stage_name="Validation Scoring"):
        data={}
        t_start = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] [{stage_name}] Scoring {len(valid)} validation entities...", flush=True)
        for st in range(0,len(valid),args.query_batch):
            stop = min(len(valid), st + args.query_batch)
            data.update(v4.score_batch(valid[st:stop],
                     vl_candidates[st:stop],store,model))
            msg = format_progress(stage_name, stop, len(valid), t_start)
            print(msg, flush=True)
            LOG.info(msg)
        return data
    baseline_scores=score_validation(baseline, stage_name="Validation Scoring (Baseline)")
    tune_base={rid:p for rid,p in baseline_scores.items() if rid in tune_ids}
    bscore,bthreshold=tune(tune_base,truth)
    LOG.info('Baseline: tuning macro F0.5=%.5f threshold=%.3f',bscore,bthreshold)
    final_model=baseline
    final_scored=baseline_scores
    final_threshold=bthreshold
    if not args.skip_mining:
        LOG.info('Mining previously unseen high-scoring negatives from ALL retrieved train candidates')
        xx,yy=mine_negatives(train,tr_candidates,truth,store,baseline,args.mining_per_entity,args.query_batch)
        LOG.info('Mined %d negative pairs',len(yy))
        if len(yy):
            X2=np.concatenate([X,xx],axis=0)
            y2=np.concatenate([y,yy],axis=0)
            del xx,yy
            gc.collect()
            print(f"[{time.strftime('%H:%M:%S')}] [STAGE 5: Refined Model] Assembling updated training array ({len(y2)} pairs)...", flush=True)
            refined=fit_xgb(X2,y2,device,850,args.cpu_threads,stage_name="Refined XGBoost (Hard Negatives)")
            del X2,y2
            gc.collect()
            refined_scores=score_validation(refined, stage_name="Validation Scoring (Refined)")
            rscore,rthreshold=tune({rid:p for rid,p in refined_scores.items() if rid in tune_ids},truth)
            LOG.info('Mined model: tuning macro F0.5=%.5f threshold=%.3f',rscore,rthreshold)
            if rscore>bscore:
                final_model=refined
                final_scored=refined_scores
                final_threshold=rthreshold
                LOG.info('Selected mined model by TUNING subset only')
            else:
                LOG.info('Kept baseline by TUNING subset only; mining did not help here')
    report_scores={rid:p for rid,p in final_scored.items() if rid in report_ids}
    report_score,report_conf=evaluate(report_scores,truth,final_threshold)
    whole_score,whole_conf=evaluate(final_scored,truth,final_threshold)
    LOG.info('HELD-OUT REPORT F0.5 = %.5f at threshold %.3f; TP/FP/FN=%s',
             report_score,final_threshold,report_conf)
    LOG.info('Full 2000-entity validation F0.5 (partly tuned) = %.5f; TP/FP/FN=%s',
             whole_score,whole_conf)
    diagnostic(final_scored,valid,vl_candidates,truth,store,final_threshold,out)
    final_model.save_model(str(model_dir/'v5_xgboost.json'))
    joblib.dump({'threshold':final_threshold,'heldout_f05':report_score,
       'full_valid_f05':whole_score,'oracle':ceiling,'settings':vars(args)},
       model_dir/'v5_metrics.joblib')
    (out/'v5_results.txt').write_text(
        f'Oracle: {ceiling:.6f}\nHeld-out report F0.5: {report_score:.6f}\n'
        f'Full validation F0.5 (partly tuned): {whole_score:.6f}\n'
        f'Threshold: {final_threshold:.4f}\nTP FP FN on held-out: {report_conf}\n',encoding='utf8')
    del X,y,baseline_scores,tr_candidates,vl_candidates
    gc.collect()
    store.close()
    if args.full_test:
        LOG.warning('Full test inference can take DAYS with extra FTS retrieval over 1.7M queries. Ensure time/disk budget.')
        args.output=out
        args.work=work  # Reuse V4 test FAISS/FTS cache if one exists.
        args.ef_search=192
        args.index_batch=10000
        args.exact_cap=70
        args.rebuild=False
        args.fts_per_probe=args.fts_per_probe
        v4.write_test(args.data,encoder,final_model,final_threshold,args)
    LOG.info('V5 pipeline completed')


if __name__=='__main__':
    main()
