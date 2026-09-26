#!/usr/bin/env python3
r"""Scale V5 training while retaining its exact original 2,000-entity validation split.

Requires model_v5_experimental.py, model_v4_hybrid.py and V4 train index caches.
Reads (but never overwrites) V5's original 10,000-sample candidate caches.
New extra-reference sampling, retrieval and feature caches are independently versioned
and resumable at the chunk level. Ground truth is used for training labels only.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path

import faiss
import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split

import model_v5_experimental as v5
import model_v4_hybrid as v4

LOG = logging.getLogger('v5_fixed_validation')
SEED = 42


def atomic_dump(obj, path):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    joblib.dump(obj, temp, compress=0)
    os.replace(temp, path)


def fingerprint(records):
    h = hashlib.sha256()
    for r in records:
        h.update(str(r[0]).encode('utf-8'))
        h.update(b'\0')
    return h.hexdigest()[:16]


def extra_reservoir(path, excluded_ids, count, seed=SEED+1):
    """Sample from records outside the original V5 split (no validation leakage)."""
    rng = np.random.default_rng(seed)
    sample = []
    eligible = 0
    for r in v4.iter_records(path):
        if r[0] in excluded_ids:
            continue
        eligible += 1
        if len(sample) < count:
            sample.append(r)
        else:
            j = int(rng.integers(eligible))
            if j < count:
                sample[j] = r
    if len(sample) < count:
        raise ValueError(f'Only {eligible} additional eligible S1 records; requested {count}')
    return sample


def score_validation(valid, candidates, store, model, query_batch):
    scored = {}
    for st in range(0, len(valid), query_batch):
        stop = min(st + query_batch, len(valid))
        scored.update(v4.score_batch(valid[st:stop], candidates[st:stop], store, model))
        LOG.info('Validation scoring %s/%s', stop, len(valid))
    return scored


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('student_resource/dataset'))
    p.add_argument('--train-size', type=int, default=50000,
                   help='TOTAL training references, including the original 8000')
    p.add_argument('--top-k', type=int, default=40)
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--svd-samples', type=int, default=60000)
    p.add_argument('--cpu-threads', type=int, default=12)
    p.add_argument('--query-batch', type=int, default=150)
    p.add_argument('--train-chunk', type=int, default=1000)
    p.add_argument('--max-candidates', type=int, default=520)
    p.add_argument('--extra-cap', type=int, default=700)
    p.add_argument('--fts-per-probe', type=int, default=55)
    p.add_argument('--extra-probe-results', type=int, default=65)
    p.add_argument('--mining-per-entity', type=int, default=10)
    p.add_argument('--skip-mining', action='store_true')
    args = p.parse_args()
    if args.train_size < 8000:
        p.error('--train-size must be >= 8000 (original V5 train set)')
    if args.train_chunk < 1 or args.query_batch < 1:
        p.error('--train-chunk and --query-batch must be positive')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    faiss.omp_set_num_threads(args.cpu_threads)
    torch.set_num_threads(args.cpu_threads)
    v4.features = v5.extra_features
    v4.TargetStore = v5.ExpandedStore
    v5.ExpandedStore.extras_per_probe = args.extra_probe_results
    v5.ExpandedStore.extra_cap = args.extra_cap
    v5.ExpandedStore.extra_enabled = True

    root = args.data.parent
    work = root / 'artifacts_v3'
    original_out = root / 'output_v5'
    output = root / 'output_v5_scaled' / f'train_{args.train_size}'
    model_dir = root / 'models_v5_scaled' / f'train_{args.train_size}'
    cache = root / 'artifacts_v5_scaled'
    for directory in (output, model_dir, cache):
        directory.mkdir(parents=True, exist_ok=True)
    ec = work / f'encoder_d{args.dim}_s{args.svd_samples}.joblib'
    ix_dir = work / f'train_indexes_d{args.dim}_s{args.svd_samples}'
    db = work / f'train_targets_d{args.dim}_s{args.svd_samples}.sqlite'
    if not all(x.exists() for x in (ec, ix_dir/'complete.joblib', db)):
        p.error('Missing V4 caches. Run original V4 first with matching encoder settings.')

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOG.info('Loading V4 encoder and cached 10M-record indexes (%s)', dev)
    encoder = joblib.load(ec)
    encoder.device = dev
    encoder.projection = torch.from_numpy(encoder.svd.components_.T.copy()).to(dev)
    store = v5.ExpandedStore(db, args.dim, ef_search=192, cache_dir=ix_dir,
                             max_candidates=args.max_candidates, exact_cap=70,
                             fts_per_probe=args.fts_per_probe)
    store.build(args.data, 'train', encoder)

    # Exactly reproduce original V5 10k reservoir sample and train_test_split.
    base = v4.sample_records(v4.source_path(args.data, 'train', 1), 10000)
    base_train, valid = train_test_split(base, test_size=.2, random_state=SEED)
    tune_refs, report_refs = train_test_split(valid, test_size=.50, random_state=SEED+1)
    tune_ids = {r[0] for r in tune_refs}
    report_ids = {r[0] for r in report_refs}
    LOG.info('Fixed original V5 split: 8000 base train, %d tune, %d report',
             len(tune_refs), len(report_refs))
    truth = v4.read_truth(args.data/'train'/'train_ground_truth.tsv')
    base_key = (f's10000_k{args.top_k}_m{args.max_candidates}_e{args.extra_cap}'
                f'_p{args.extra_probe_results}_x1')
    trp = original_out/f'train_candidates_{base_key}.joblib'
    vap = original_out/f'valid_candidates_{base_key}.joblib'
    if not trp.exists() or not vap.exists():
        p.error(f'Original V5 caches are needed to avoid repeating the first 10K retrieval. '
                f'Expected {trp} and {vap}. Use EXACT original V5 retrieval settings.')
    LOG.info('Loading original V5 base/validation candidate caches')
    base_candidates = joblib.load(trp)
    valid_candidates = joblib.load(vap)
    if len(base_candidates) != len(base_train) or len(valid_candidates) != len(valid):
        raise ValueError('Original V5 cache length mismatch: check the sampling seed/configuration')
    LOG.info('Fixed validation retrieval recall:')
    v4.candidate_recall(valid, valid_candidates, truth)
    oracle = v4.retrieval_oracle_f05(valid, valid_candidates, truth)

    needed = args.train_size - len(base_train)
    sampling_cache = cache/f'extra_refs_{needed}_seed43_excluding_orig10000.joblib'
    if sampling_cache.exists():
        LOG.info('Loading sampled extra records from %s', sampling_cache)
        extra = joblib.load(sampling_cache)
    else:
        LOG.info('Sampling %d additional S1 records, EXCLUDING all original 10K (including validation)', needed)
        extra = extra_reservoir(v4.source_path(args.data,'train',1),
                                {r[0] for r in base}, needed)
        atomic_dump(extra, sampling_cache)
    if len(extra) != needed or {r[0] for r in extra} & {r[0] for r in base}:
        raise ValueError('Additional training sample has duplicates or validation leakage')
    LOG.info('Scaling experiment: %d total training S1 and unchanged %d validation S1',
             len(base_train)+len(extra), len(valid))

    config = f'k{args.top_k}_m{args.max_candidates}_e{args.extra_cap}_p{args.extra_probe_results}_fts{args.fts_per_probe}'
    feature_cache = cache/f'features_{config}'
    feature_cache.mkdir(exist_ok=True)
    candidate_cache = cache/f'candidates_{config}'
    candidate_cache.mkdir(exist_ok=True)

    def extra_blocks():
        for start in range(0,len(extra),args.train_chunk):
            block = extra[start:start+args.train_chunk]
            key = f'{fingerprint(block)}_{len(block)}'
            candidate_file = candidate_cache/f'{key}.joblib'
            if candidate_file.exists():
                candidates = joblib.load(candidate_file)
                if len(candidates) != len(block):
                    raise ValueError(f'Corrupt candidate cache {candidate_file}')
                LOG.info('Extra retrieval cache loaded %d/%d', min(start+len(block), len(extra)),len(extra))
            else:
                candidates = []
                for pos in range(0,len(block),args.query_batch):
                    candidates.extend(store.retrieve(block[pos:pos+args.query_batch],encoder,args.top_k))
                atomic_dump(candidates, candidate_file)
                LOG.info('Extra retrieval cached %d/%d', min(start+len(block),len(extra)),len(extra))
            yield block, candidates, key

    # Reuses feature matrices across reruns. Cache each chunk before continuing.
    base_feature_file = feature_cache/f'base_{fingerprint(base_train)}.joblib'
    if base_feature_file.exists():
        X_base,y_base = joblib.load(base_feature_file)
    else:
        X_base,y_base = v4.training_arrays(base_train,base_candidates,truth,store,max_neg=24)
        atomic_dump((X_base,y_base),base_feature_file)
    xs=[X_base]
    ys=[y_base]
    for block,candidates,key in extra_blocks():
        featfile=feature_cache/f'{key}.joblib'
        if featfile.exists():
            xx,yy=joblib.load(featfile)
        else:
            xx,yy=v4.training_arrays(block,candidates,truth,store,max_neg=24)
            atomic_dump((xx,yy),featfile)
        xs.append(xx)
        ys.append(yy)
        del candidates
    X=np.concatenate(xs)
    y=np.concatenate(ys)
    del xs,ys,X_base,y_base
    gc.collect()
    LOG.info('Initial training pairs %d; positives=%d; negatives=%d',len(y),int(y.sum()),int((y==0).sum()))
    device='cuda' if dev.type=='cuda' else 'cpu'
    baseline=v5.fit_xgb(X,y,device,650,args.cpu_threads)
    baseline_scored=score_validation(valid,valid_candidates,store,baseline,args.query_batch)
    base_tune,base_threshold=v5.tune({rid:pp for rid,pp in baseline_scored.items() if rid in tune_ids},truth)
    LOG.info('Scaled baseline: tuning F0.5=%.5f threshold=%.3f',base_tune,base_threshold)
    best_model=baseline
    best_scored=baseline_scored
    best_threshold=base_threshold
    if not args.skip_mining:
        LOG.info('Mining additional negatives from original 8K and extra %d S1',len(extra))
        neg_x=[]
        neg_y=[]
        bx,by=v5.mine_negatives(base_train,base_candidates,truth,store,baseline,
                                args.mining_per_entity,args.query_batch)
        neg_x.append(bx);neg_y.append(by)
        for block,candidates,key in extra_blocks():
            # Must re-mine against the *current* baseline; never reuse stale mined examples.
            xx,yy=v5.mine_negatives(block,candidates,truth,store,baseline,
                                     args.mining_per_entity,args.query_batch)
            neg_x.append(xx);neg_y.append(yy)
            del candidates
        X2=np.concatenate([X,*neg_x]);y2=np.concatenate([y,*neg_y])
        del neg_x,neg_y
        LOG.info('Training refined model with %d pairs',len(y2))
        refined=v5.fit_xgb(X2,y2,device,850,args.cpu_threads)
        del X2,y2
        gc.collect()
        refined_scored=score_validation(valid,valid_candidates,store,refined,args.query_batch)
        mined_tune,mined_threshold=v5.tune({rid:pp for rid,pp in refined_scored.items() if rid in tune_ids},truth)
        LOG.info('Scaled mined: tuning F0.5=%.5f threshold=%.3f',mined_tune,mined_threshold)
        if mined_tune>base_tune:
            best_model,best_scored,best_threshold=refined,refined_scored,mined_threshold
    report_score,report_conf=v5.evaluate({rid:pp for rid,pp in best_scored.items() if rid in report_ids},truth,best_threshold)
    whole_score,whole_conf=v5.evaluate(best_scored,truth,best_threshold)
    LOG.info('FIXED REPORT F0.5=%.5f at threshold %.3f TP/FP/FN=%s',report_score,best_threshold,report_conf)
    LOG.info('FIXED FULL VALID F0.5=%.5f (partly tuned) TP/FP/FN=%s',whole_score,whole_conf)
    v5.diagnostic(best_scored,valid,valid_candidates,truth,store,best_threshold,output)
    best_model.save_model(str(model_dir/'v5_scaled_xgboost.json'))
    (output/'scaled_results.txt').write_text(
        f'Train S1: {args.train_size}\nOriginal fixed validation S1: {len(valid)}\n'
        f'Oracle: {oracle:.6f}\nFixed report F0.5: {report_score:.6f}\n'
        f'Full valid F0.5 partly tuned: {whole_score:.6f}\nThreshold: {best_threshold:.4f}\n'
        f'Fixed report TP/FP/FN: {report_conf}\n',encoding='utf8')
    joblib.dump({'threshold':best_threshold,'train_size':args.train_size,
                 'report_score':report_score,'oracle':oracle,'settings':vars(args)},
                 model_dir/'v5_scaled_metrics.joblib')
    LOG.info('Experiment completed. Result: %s',output/'scaled_results.txt')
    store.close()


if __name__=='__main__':
    main()
