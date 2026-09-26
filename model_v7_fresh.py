#!/usr/bin/env python3
r"""V7 fresh retraining: combined + independent name/address FAISS retrieval.

Requires model_v4_hybrid.py, model_v5_experimental.py and model_v6_fast.py
beside this file. Never uses validation truth to generate candidates or features.
The independent country/source indexes are saved separately for manageable RAM.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path

import faiss
import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

import model_v4_hybrid as v4
import model_v5_experimental as v5
import model_v6_fast as v6

LOG = logging.getLogger('v7')
SEED = 2026  # NEW held-out split, not V2-V6's repeatedly inspected split


def field_text(field, r):
    return v4.name_key(r[1]) if field == 'name' else v4.addr_key(r[2])


class FieldEncoder(v4.Encoder):
    """Distinct fitted TF-IDF/SVD per field; never patches shared V4 globals."""
    def __init__(self, field, dim, device):
        super().__init__(dim, device)
        self.field = field

    def fit_samples(self, samples):
        texts = [field_text(self.field, r) for r in samples]
        LOG.info('Fitting %s TF-IDF/SVD on %d sampled records', self.field, len(texts))
        matrix = self.vectorizer.fit_transform(texts)
        self.svd.fit(matrix)
        del matrix, texts
        gc.collect()
        self.projection = torch.from_numpy(self.svd.components_.T.copy()).to(self.device)

    @torch.no_grad()
    def transform(self, records, batch_size=2000):
        output = []
        for start in range(0, len(records), batch_size):
            part = records[start:start + batch_size]
            sparse = self.vectorizer.transform([field_text(self.field, r) for r in part]).tocoo()
            if self.device.type == 'cuda':
                indices = torch.from_numpy(np.stack((sparse.row, sparse.col)).astype(np.int64)).to(self.device)
                values = torch.from_numpy(sparse.data.astype(np.float32, copy=False)).to(self.device)
                st = torch.sparse_coo_tensor(indices, values, sparse.shape, device=self.device,
                                             check_invariants=False).coalesce()
                dense = torch.nn.functional.normalize(torch.sparse.mm(st, self.projection).float(), dim=1)
                arr = dense.cpu().numpy()
                del indices, values, st, dense
            else:
                arr = normalize(sparse.tocsr() @ self.svd.components_.T).astype(np.float32)
            output.append(np.ascontiguousarray(arr, dtype=np.float32))
        return np.concatenate(output) if output else np.empty((0, self.dim), np.float32)


def save_encoder(encoder, path):
    proj = encoder.projection
    encoder.projection = None
    tmp = path.with_suffix('.tmp')
    joblib.dump(encoder, tmp, compress=0)
    tmp.replace(path)
    encoder.projection = proj


def load_encoder(path, device):
    obj = joblib.load(path)
    obj.device = device
    obj.projection = torch.from_numpy(obj.svd.components_.T.copy()).to(device)
    return obj


def build_encoders(args, work, device):
    files = {field: work / f'encoder_{field}_d{args.dim}_s{args.svd_samples}.joblib'
             for field in ('combined', 'name', 'addr')}
    if all(p.exists() for p in files.values()):
        LOG.info('Loading all three V7 fitted encoders')
        return {k: load_encoder(p, device) for k, p in files.items()}
    LOG.info('Sampling training records ONCE for three independent encoders')
    paths = [v4.source_path(args.data, 'train', i) for i in (1, 2, 3)]
    samples = []
    for j, p in enumerate(paths):
        samples.extend(v4.sample_records(p, args.svd_samples, seed=SEED + 101 + j))
    result = {}
    for field in ('combined', 'name', 'addr'):
        if files[field].exists():
            result[field] = load_encoder(files[field], device)
            continue
        if field == 'combined':
            obj = v4.Encoder(args.dim, device)
            LOG.info('Fitting combined TF-IDF/SVD on %d sampled records', len(samples))
            texts = [v4.text_of(r) for r in samples]
            matrix = obj.vectorizer.fit_transform(texts)
            obj.svd.fit(matrix)
            del matrix, texts
            gc.collect()
            obj.projection = torch.from_numpy(obj.svd.components_.T.copy()).to(device)
        else:
            obj = FieldEncoder(field, args.dim, device)
            obj.fit_samples(samples)
        save_encoder(obj, files[field])
        result[field] = obj
        LOG.info('Saved fresh V7 %s encoder', field)
    return result


def build_combined(args, work, encoder):
    folder = work / 'combined_indexes'
    db = work / 'train_targets.sqlite'
    complete = folder / 'complete.joblib'
    if complete.exists() and db.exists():
        LOG.info('Fresh V7 combined indexes already complete on disk')
        return db
    LOG.info('Building FRESH combined V7 FAISS indexes (10M targets)')
    store = v4.TargetStore(db, args.dim, m=args.hnsw_m, ef_search=args.ef_search,
                           cache_dir=folder, rebuild=True, max_candidates=args.base_cap,
                           fts_per_probe=args.fts_per_probe)
    store.build(args.data, 'train', encoder, chunk_size=args.index_batch)
    store.indexes.clear()
    store.ids.clear()
    store.close()
    del store
    gc.collect()
    return db


def build_field_indexes(args, work, encoders):
    """Build name/address HNSW indexes one SOURCE at a time to bound memory."""
    folder = work / 'field_indexes'
    folder.mkdir(parents=True, exist_ok=True)
    for source in (2, 3):
        marker = folder / f'source_{source}.complete.joblib'
        if marker.exists():
            LOG.info('Source %d independent field indexes complete', source)
            continue
        LOG.info('Building fresh NAME and ADDRESS FAISS indexes for Source %d', source)
        indexes = {}
        ids = defaultdict(list)
        n = 0
        def ingest(chunk):
            groups = defaultdict(list)
            for i, rec in enumerate(chunk):
                groups[rec[3]].append(i)
            # Transform once per batch per field; group by country after encoding.
            embeddings = {field: encoders[field].transform(chunk, batch_size=1500)
                          for field in ('name', 'addr')}
            for country, positions in groups.items():
                key = (source, country)
                if ('name', source, country) not in indexes:
                    for field in ('name', 'addr'):
                        ix = faiss.IndexHNSWFlat(args.dim, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
                        ix.hnsw.efConstruction = args.ef_construction
                        ix.hnsw.efSearch = args.ef_search
                        indexes[(field, source, country)] = ix
                    LOG.info('Separate indexes created for (%s, %s)', source, country)
                for field in ('name', 'addr'):
                    indexes[(field, source, country)].add(
                        np.ascontiguousarray(embeddings[field][positions], dtype=np.float32))
                ids[key].extend(chunk[i][0] for i in positions)
        chunk = []
        for rec in v4.iter_records(v4.source_path(args.data, 'train', source)):
            chunk.append(rec)
            if len(chunk) >= args.index_batch:
                ingest(chunk)
                n += len(chunk)
                if n % 200000 < args.index_batch:
                    LOG.info('Source %d independent name/address indexed %d', source, n)
                chunk.clear()
        if chunk:
            ingest(chunk)
            n += len(chunk)
        # Save and verify both channels have IDs in exactly identical order.
        keys = sorted(ids)
        for key in keys:
            src, country = key
            tag = f'{src}_{country}'
            for field in ('name', 'addr'):
                ix = indexes[(field, src, country)]
                if ix.ntotal != len(ids[key]):
                    raise RuntimeError(f'{field} index-ID alignment failure for {tag}')
                faiss.write_index(ix, str(folder / f'{field}_{tag}.faiss'))
            joblib.dump(ids[key], folder / f'{tag}.ids.joblib', compress=0)
            LOG.info('Saved independent FAISS: %s (%d records)', tag, len(ids[key]))
        joblib.dump({'source':source, 'keys':keys, 'dim':args.dim,
                     'hnsw_m':args.hnsw_m, 'encoder_samples':args.svd_samples}, marker)
        del indexes, ids, chunk
        gc.collect()
        LOG.info('Source %d independent channels complete: %d records', source, n)


class GroupRetriever:
    """Country-at-a-time loading limits peak RAM versus 10M x 3 resident indexes."""
    def __init__(self, args, work, encoders, db):
        self.args = args
        self.work = work
        self.encoders = encoders
        self.store = v5.ExpandedStore(db, args.dim, cache_dir=None,
                                      ef_search=args.ef_search, max_candidates=args.base_cap,
                                      exact_cap=args.exact_cap, fts_per_probe=args.fts_per_probe)
        self.store.extra_cap = args.v5_cap
        self.store.extras_per_probe = args.extra_probe_results
        self.store.extra_enabled = True
        self.fields = {}
        self.ids = {}
        self.country = None

    def activate(self, country):
        if country == self.country:
            return
        self.store.indexes.clear()
        self.store.ids.clear()
        self.fields.clear()
        self.ids.clear()
        gc.collect()
        cdir = self.work / 'combined_indexes'
        fdir = self.work / 'field_indexes'
        for source in (2, 3):
            key = (source, country)
            tag = f'{source}_{country}'
            cp = cdir / f'{tag}.faiss'
            fp = fdir / f'{tag}.ids.joblib'
            if not cp.exists() or not fp.exists():
                continue
            cix = faiss.read_index(str(cp))
            cix.hnsw.efSearch = self.args.ef_search
            self.store.indexes[key] = cix
            self.store.ids[key] = joblib.load(cdir / f'{tag}.ids.joblib')
            self.ids[key] = joblib.load(fp)
            for field in ('name', 'addr'):
                ix = faiss.read_index(str(fdir / f'{field}_{tag}.faiss'))
                ix.hnsw.efSearch = self.args.ef_search
                if ix.ntotal != len(self.ids[key]):
                    raise RuntimeError(f'{field} index-ID mismatch for {tag}')
                self.fields[(field, source)] = ix
            LOG.info('Loaded fresh V7 combined/name/address indexes: source %d country %s', source, country)
        self.country = country

    def retrieve(self, records):
        if not records:
            return []
        country = records[0][3]
        if any(r[3] != country for r in records):
            raise ValueError('GroupRetriever.retrieve expects a single-country batch')
        self.activate(country)
        # Use V5's full expanded FTS + exact + combined FAISS routes, freshly generated.
        base = self.store.retrieve(records, self.encoders['combined'], self.args.top_k)
        extra = {'name': [[] for _ in records], 'addr': [[] for _ in records]}
        for field in ('name', 'addr'):
            q = self.encoders[field].transform(records)
            for source in (2, 3):
                ix = self.fields.get((field, source))
                if ix is None or ix.ntotal == 0:
                    continue
                scores, labels = ix.search(q, min(self.args.field_top_k, ix.ntotal))
                identities = self.ids[(source, country)]
                for i, line in enumerate(labels):
                    extra[field][i].extend(identities[int(label)] for label in line if label >= 0)
        merged = []
        for i, old in enumerate(base):
            found = {cid for cid, _ in old}
            row = list(old)
            # Equal representation for independent name/address ANN routes.
            na, aa = extra['name'][i], extra['addr'][i]
            for j in range(max(len(na), len(aa))):
                for route in (na, aa):
                    if j < len(route) and route[j] not in found:
                        cid = route[j]
                        row.append((cid, -1.0))  # Unknown combined cosine; NOT a negative similarity.
                        found.add(cid)
                if len(row) >= self.args.total_cap:
                    break
            merged.append(row[:self.args.total_cap])
        return merged

    def close(self):
        self.store.close()
        self.store.indexes.clear()
        self.store.ids.clear()
        self.fields.clear()
        self.ids.clear()


def selected_truth(path, ids):
    """Avoid V4's full 2.2M-row truth dictionary in constrained 32GB RAM."""
    output = {}
    with path.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for line in reader:
            rid = line['source1_entity_id']
            if rid in ids:
                output[rid] = set(filter(None, line['matched_entity_ids'].split(',')))
    for rid in ids:
        output.setdefault(rid, set())
    return output


def candidates_fresh(split_name, records, retriever, args, out):
    """Fresh retrieval, resumable PER COUNTRY and per chunk, never reads V4/V5 caches."""
    config = {k:getattr(args,k) for k in ('sample','top_k','field_top_k','dim',
             'base_cap','v5_cap','total_cap','exact_cap','fts_per_probe',
             'extra_probe_results','ef_search')}
    tag = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    folder = out / f'{split_name}_candidate_chunks_{tag}'
    folder.mkdir(parents=True, exist_ok=True)
    answer = [None] * len(records)
    country_indices = defaultdict(list)
    for i, rec in enumerate(records):
        country_indices[rec[3]].append(i)
    for country in sorted(country_indices):
        idxs = country_indices[country]
        LOG.info('Retrieving %s fresh candidates, country=%s, S1=%d', split_name, country, len(idxs))
        for st in range(0, len(idxs), args.query_batch):
            positions = idxs[st:st + args.query_batch]
            cp = folder / f'{country}_{st:07d}.joblib'
            if cp.exists() and not args.recompute_candidates:
                got = joblib.load(cp)
                if len(got) != len(positions):
                    raise RuntimeError(f'Candidate chunk mismatch: {cp}')
            else:
                got = retriever.retrieve([records[i] for i in positions])
                tmp = cp.with_suffix('.tmp')
                joblib.dump(got, tmp, compress=0)
                tmp.replace(cp)
            for i, row in zip(positions, got):
                answer[i] = row
            LOG.info('%s retrieval %d/%d S1 in %s', split_name,
                     min(st + len(positions), len(idxs)), len(idxs), country)
    if any(row is None for row in answer):
        raise RuntimeError('Incomplete fresh candidate generation')
    return answer


def score_validation(records, candidates, store, model, args):
    scores = {}
    for st in range(0, len(records), args.score_batch):
        p = v4.score_batch(records[st:st + args.score_batch],
                           candidates[st:st + args.score_batch], store, model)
        scores.update(p)
        LOG.info('Validation scoring %d/%d', min(st + args.score_batch, len(records)), len(records))
    return scores


def threshold_report(scored, tune_ids, report_ids, truth):
    tuning = {rid: scored[rid] for rid in tune_ids}
    for t in np.arange(.30, .991, .025):
        score, _ = v5.evaluate(tuning, truth, float(t))
        LOG.info('Threshold %.3f | Tuning Macro F0.5 %.5f', t, score)
    fine = np.unique(np.r_[np.arange(.25, .991, .025), np.arange(.40, .981, .005)])
    val, threshold = max(((v5.evaluate(tuning, truth, float(t))[0], float(t))
                          for t in fine), key=lambda item: item[0])
    held, conf = v5.evaluate({rid:scored[rid] for rid in report_ids}, truth, threshold)
    LOG.info('Best tuning F0.5: %.5f', val)
    LOG.info('Selected threshold: %.3f', threshold)
    LOG.info('HELD-OUT REPORT F0.5 = %.5f at threshold %.3f; TP/FP/FN=%s', held, threshold, conf)
    return val, held, threshold, conf


def main():
    p = argparse.ArgumentParser(description='V7 fully fresh combined/name/address retrieval + supervised retraining')
    p.add_argument('--data', type=Path, default=Path('student_resource/dataset'))
    p.add_argument('--sample', type=int, default=10000)
    p.add_argument('--top-k', type=int, default=40)
    p.add_argument('--field-top-k', type=int, default=40)
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--svd-samples', type=int, default=60000)
    p.add_argument('--cpu-threads', type=int, default=12)
    p.add_argument('--query-batch', type=int, default=75)
    p.add_argument('--score-batch', type=int, default=50)
    p.add_argument('--index-batch', type=int, default=10000)
    p.add_argument('--hnsw-m', type=int, default=16)
    p.add_argument('--ef-construction', type=int, default=100)
    p.add_argument('--ef-search', type=int, default=192)
    p.add_argument('--base-cap', type=int, default=520)
    p.add_argument('--v5-cap', type=int, default=700)
    p.add_argument('--total-cap', type=int, default=950)
    p.add_argument('--exact-cap', type=int, default=70)
    p.add_argument('--fts-per-probe', type=int, default=55)
    p.add_argument('--extra-probe-results', type=int, default=65)
    p.add_argument('--mining-batch', type=int, default=40)
    p.add_argument('--feature-workers', type=int, default=2)
    p.add_argument('--mining-per-entity', type=int, default=10)
    p.add_argument('--skip-mining', action='store_true')
    p.add_argument('--retrieval-only', action='store_true')
    p.add_argument('--skip-test', action='store_true', help='Accepted for V4-style CLI; V7 is validation-only')
    p.add_argument('--rebuild', action='store_true', help='Delete ONLY V7 artifacts and outputs, rebuild everything')
    p.add_argument('--recompute-candidates', action='store_true')
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    faiss.omp_set_num_threads(args.cpu_threads)
    torch.set_num_threads(args.cpu_threads)
    if args.query_batch < 1 or args.score_batch < 1 or args.feature_workers < 1:
        p.error('batch sizes and feature-workers must be positive')
    work = args.data.parent / 'artifacts_v7_fresh'
    out = args.data.parent / 'output_v7'
    models = args.data.parent / 'models_v7'
    if args.rebuild:
        for folder in (work, out, models):
            if folder.exists():
                LOG.warning('Deleting V7-only directory for --rebuild: %s', folder)
                shutil.rmtree(folder)
    for folder in (work, out, models):
        folder.mkdir(parents=True, exist_ok=True)
    signature = {'dim':args.dim, 'svd_samples':args.svd_samples, 'hnsw_m':args.hnsw_m,
                 'sample_seed':SEED, 'algorithm':'v7_fresh_1',
                 'training_data':[(str(v4.source_path(args.data,'train',i)),
                        v4.source_path(args.data,'train',i).stat().st_size,
                        v4.source_path(args.data,'train',i).stat().st_mtime_ns)
                        for i in (1,2,3)]}
    signature_file = work / 'build_signature.json'
    if signature_file.exists():
        prev = json.loads(signature_file.read_text())
        if prev != signature:
            p.error('V7 build settings changed; rerun with --rebuild to avoid stale encoders/indexes')
    else:
        signature_file.write_text(json.dumps(signature, indent=2))
    cuda = bool(torch.cuda.is_available())
    device = torch.device('cuda' if cuda and not args.cpu else 'cpu')
    LOG.info('PyTorch device: %s', device)
    LOG.info('CUDA available: %s', cuda)
    if device.type == 'cuda':
        LOG.info('GPU: %s', torch.cuda.get_device_name(0))
        LOG.info('VRAM: %.2f GiB', torch.cuda.get_device_properties(0).total_memory / 2**30)
    LOG.info('Training THREE independent semantic encoders (combined, name-only, address-only)')
    encoders = build_encoders(args, work, device)
    LOG.info('Sampling %d fresh training references (seed=%d)', args.sample, SEED)
    sampled = v4.sample_records(v4.source_path(args.data, 'train', 1), args.sample, seed=SEED)
    train, valid = train_test_split(sampled, test_size=.2, random_state=SEED)
    # New train/valid split; tune/report halves must not enter feature design.
    tune, report = train_test_split(valid, test_size=.5, random_state=SEED + 1)
    tune_ids = {r[0] for r in tune}
    report_ids = {r[0] for r in report}
    truth = selected_truth(args.data / 'train' / 'train_ground_truth.tsv',
                           {r[0] for r in sampled})
    LOG.info('Building three fresh training retrieval channels over full S2/S3 targets')
    db = build_combined(args, work, encoders['combined'])
    build_field_indexes(args, work, encoders)
    LOG.info('All V7 indexes built and saved; independent country loading enabled')
    retriever = GroupRetriever(args, work, encoders, db)
    train_c = candidates_fresh('train', train, retriever, args, out)
    v4.candidate_recall(train, train_c, truth)
    v4.retrieval_diagnostics(train, train_c, truth, retriever.store,
                              out / 'train_retrieval_misses_v7.tsv', 300)
    valid_c = candidates_fresh('validation', valid, retriever, args, out)
    v4.candidate_recall(valid, valid_c, truth)
    ceiling = v4.retrieval_oracle_f05(valid, valid_c, truth)
    v4.retrieval_diagnostics(valid, valid_c, truth, retriever.store,
                              out / 'validation_retrieval_misses_v7.tsv', 300)
    LOG.info('V7 idealized retrieval ceiling %.5f (no label-informed retrieval)', ceiling)
    if args.retrieval_only:
        LOG.info('Retrieval-only experiment finished; classifier not trained')
        retriever.close()
        return
    # Free ~10M x 3 FAISS index RAM before XGBoost training/scoring.
    retriever.close()
    del retriever
    gc.collect()
    store = v4.TargetStore(db, args.dim, cache_dir=None)
    # V6's feature function does NOT require saved validation audit labels.
    v4.features = v6.extra_features
    LOG.info('Generating fresh V7 training features')
    X, y = v4.training_arrays(train, train_c, truth, store, max_neg=24)
    weights = v6.positive_sample_weights(X, y, 1.35)
    LOG.info('Initial training pairs: %d (positives %d, negatives %d)',
             len(y), int(y.sum()), int((y == 0).sum()))
    device_name = 'cuda' if device.type == 'cuda' else 'cpu'
    LOG.info('Training initial XGBoost on %s', device_name)
    baseline = v6.fit_xgb(X, y, device_name, 650, args.cpu_threads, sample_weight=weights)
    baseline_scored = score_validation(valid, valid_c, store, baseline, args)
    b_tune, b_held, bthr, _ = threshold_report(baseline_scored, tune_ids, report_ids, truth)
    final_model, final_scored, final_threshold, best_tune = baseline, baseline_scored, bthr, b_tune
    selected = 'baseline'
    if not args.skip_mining:
        LOG.info('Mining previously unseen high-scoring negatives from ALL fresh V7 train candidates')
        xx, yy = v6.mine_negatives_fast(train, train_c, truth, store, baseline,
                                        args.mining_per_entity, args.mining_batch, args.feature_workers)
        LOG.info('Mined %d negative pairs', len(yy))
        if len(yy):
            X2 = np.concatenate((X, xx))
            y2 = np.concatenate((y, yy))
            w2 = np.concatenate((weights, np.ones(len(yy), dtype=np.float32)))
            del xx, yy
            gc.collect()
            LOG.info('Training mined XGBoost on %s', device_name)
            mined = v6.fit_xgb(X2, y2, device_name, 850, args.cpu_threads, sample_weight=w2)
            del X2, y2, w2
            gc.collect()
            mined_scored = score_validation(valid, valid_c, store, mined, args)
            m_tune, m_held, mthr, _ = threshold_report(mined_scored, tune_ids, report_ids, truth)
            LOG.info('Mined: tuning macro F0.5=%.5f; report macro F0.5=%.5f', m_tune, m_held)
            if m_tune > best_tune:
                final_model, final_scored, final_threshold, best_tune = mined, mined_scored, mthr, m_tune
                selected = 'mined'
                LOG.info('Selected mined model using TUNING subset only')
    selected_scores = {rid:final_scored[rid] for rid in report_ids}
    heldout, confusion = v5.evaluate(selected_scores, truth, final_threshold)
    whole, whole_conf = v5.evaluate(final_scored, truth, final_threshold)
    LOG.info('FINAL V7 HELD-OUT REPORT Macro F0.5 = %.5f', heldout)
    LOG.info('Full validation (partly tuned) Macro F0.5 = %.5f', whole)
    LOG.info('Selected threshold: %.3f; TP/FP/FN report=%s', final_threshold, confusion)
    LOG.info('Best validation F0.5: %.5f (held-out report)', heldout)
    v5.diagnostic(final_scored, valid, valid_c, truth, store, final_threshold, out)
    final_model.save_model(str(models / 'v7_selected_xgboost.json'))
    (out / 'v7_results.txt').write_text(
        f'Retrieval oracle: {ceiling:.6f}\nSelected model: {selected}\n'
        f'Tuning F0.5: {best_tune:.6f}\nHeld-out report F0.5: {heldout:.6f}\n'
        f'Full validation (partly tuned) F0.5: {whole:.6f}\n'
        f'Selected threshold: {final_threshold:.4f}\n'
        f'HELD-OUT TP FP FN: {confusion}\n', encoding='utf-8')
    joblib.dump({'threshold':final_threshold, 'oracle':ceiling, 'heldout_f05':heldout,
                 'full_validation_f05':whole, 'chosen_model':selected,
                 'fresh_seed':SEED, 'settings':vars(args)}, models / 'v7_metrics.joblib')
    del X, y, baseline_scored, train_c, valid_c
    store.close()
    LOG.info('Pipeline completed successfully.')


if __name__ == '__main__':
    main()
