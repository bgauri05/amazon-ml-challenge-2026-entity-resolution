#!/usr/bin/env python3
"""V12: targeted, validated ANN expansion of an existing complete V10 submission.

NO fitting, NO index construction, NO new external data. Reuses:
  - V10 submission and fresh validation scores
  - finished V7 train / V8 test FAISS indexes and DBs
  - V7 fitted encoders and V8 saved classifiers

Adds only new, classifier-accepted IDs for S1 records with an absent target
source, using wider ANN retrieval and high-confidence S2<->S3 anchors.
Measures changes on existing local validation data BEFORE modifying test output.
Uses a time budget; preserves the complete V10 output on errors/timeouts.

Caveats: inspected local validation is not pristine; no France training labels;
there is no basis for promising a specific hidden-test score.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import sys
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import faiss
import joblib
import numpy as np
import torch
from rapidfuzz import fuzz
import model_v4_hybrid as v4
import model_v6_fast as v6
import model_v7_fresh as v7

LOG = logging.getLogger('v12')
TARGET_HEADERS = (b'source1_entity_id\tmatched_entity_ids', b'source1_entity_id\tcandidate_entity_ids')


def safe_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ids(cell):
    return list(dict.fromkeys(x for x in cell.split(',') if x)) if cell else []


def metric(p, y):
    if not y:
        return float(not p)
    return 1.25 * len(p & y) / (.25 * len(y) + len(p))


def source(cid):
    if cid.startswith('S2-'):
        return 2
    if cid.startswith('S3-'):
        return 3
    raise ValueError('Invalid target ID ' + str(cid))


def load_models(folder, threads):
    meta = json.loads((folder / 'v8_metadata.json').read_text(encoding='utf8'))
    selected = meta['selected']
    if selected == 'v7_original':
        filenames, weight = ['v8_selected_v7.json'], 1.
    elif selected == 'precision':
        filenames, weight = ['v8_precision.json'], 1.
    elif selected == 'recall':
        filenames, weight = ['v8_recall.json'], 1.
    elif selected.startswith('blend_'):
        filenames, weight = ['v8_precision.json', 'v8_recall.json'], float(selected[6:]) / 100.
    else:
        raise RuntimeError('Unknown selected V8 model: ' + selected)
    import xgboost as xgb
    models = []
    for name in filenames:
        model = xgb.XGBClassifier()
        model.load_model(str(folder / name))
        model.set_params(device='cpu', n_jobs=threads)
        models.append(model)
    return models, weight


def load_encoders(folder, device):
    # These exact classes were stored under __main__ by the original V7 run.
    import __main__
    __main__.FieldEncoder = v7.FieldEncoder
    __main__.Encoder = v4.Encoder
    enc = {}
    for field in ('combined', 'name', 'addr'):
        path = folder / f'encoder_{field}_d64_s60000.joblib'
        if not path.exists():
            raise FileNotFoundError(str(path))
        enc[field] = v7.load_encoder(path, device)
        LOG.info('Loaded existing %s encoder', field)
    return enc


def normalize_similarity(a, b):
    na, nb = v4.name_key(a[1]), v4.name_key(b[1])
    ad, bd = v4.addr_key(a[2]), v4.addr_key(b[2])
    n = fuzz.WRatio(na, nb) / 100. if na and nb else 0.
    addr = fuzz.WRatio(ad, bd) / 100. if ad and bd else 0.
    x, y = set(re.findall(r'\d+', ad)), set(re.findall(r'\d+', bd))
    numbers_ok = not (x and y and x.isdisjoint(y))
    return n, addr, numbers_ok


class Expander:
    def __init__(self, a, work):
        self.a = a
        self.work = work
        db = work / 'train_targets.sqlite'
        if not db.exists():
            raise FileNotFoundError(str(db))
        self.db = sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True)
        self.db.execute('PRAGMA query_only=ON')
        self.db.execute('PRAGMA cache_size=-65536')
        self.db.execute('PRAGMA mmap_size=1073741824')
        self.models, self.weight = load_models(a.root / 'models_v8', a.threads)
        device = torch.device('cuda' if a.cuda and torch.cuda.is_available() else 'cpu')
        self.enc = load_encoders(a.root / 'artifacts_v7_fresh', device)
        self.country = None
        self.indexes = {}
        self.keycache = {}
        self.total_queries = 0
        self.total_features = 0

    def get(self, keys):
        new = list(dict.fromkeys(k for k in keys if k and k not in self.keycache))
        for i in range(0, len(new), 380):
            part = new[i:i+380]
            sql = 'SELECT id,name,addr,country FROM targets WHERE id IN (' + ','.join('?'*len(part)) + ')'
            self.keycache.update({r[0]:r for r in self.db.execute(sql, part)})
        # Bounded across batches: avoid caching millions of Python tuples.
        if len(self.keycache) > 180000:
            wanted = set(keys)
            self.keycache = {k:v for k,v in self.keycache.items() if k in wanted}
        return self.keycache

    def activate(self, country, sources):
        if country != self.country:
            self.indexes.clear()
            self.keycache.clear()
            gc.collect()
            self.country = country
        for src in sources:
            for field in ('combined', 'name', 'addr'):
                key = (src, field)
                if key in self.indexes:
                    continue
                folder = self.work / ('combined_indexes' if field == 'combined' else 'field_indexes')
                tag = f'{src}_{country}'
                ix_file = folder / (f'{tag}.faiss' if field == 'combined' else f'{field}_{tag}.faiss')
                ids_file = folder / f'{tag}.ids.joblib'
                if not ix_file.exists() or not ids_file.exists():
                    raise FileNotFoundError(f'Existing ANN index missing: {ix_file} / {ids_file}')
                ix = faiss.read_index(str(ix_file))
                if hasattr(ix, 'hnsw'):
                    ix.hnsw.efSearch = self.a.ef
                lookup = joblib.load(ids_file)
                if len(lookup) != ix.ntotal:
                    raise RuntimeError('ANN index/ID alignment error: ' + str(ix_file))
                self.indexes[key] = (ix, lookup)
                LOG.info('Loaded %s source %s country %s (%s records)', field, src, country, ix.ntotal)

    def expand(self, rows, predicted, seen):
        if not rows:
            return []
        country = rows[0][3]
        if any(r[3] != country for r in rows):
            raise ValueError('A batch must contain one country')
        needed = []
        need_anchor_rows = []
        for mids in predicted:
            found_sources = {source(k) for k in mids if k.startswith(('S2-', 'S3-'))}
            need_anchor_rows.append(found_sources != {2, 3})
        if not any(need_anchor_rows):
            return [[] for _ in rows]
        predicted_keys = [k for take, group in zip(need_anchor_rows, predicted) if take for k in group]
        self.get(predicted_keys)
        # A row with matches in both sources is intentionally left unchanged.
        # The broad ANN search targets only currently missing sources.
        queries = defaultdict(list)  # target source -> (entity index, seed, seed_type)
        for i, (row, mids) in enumerate(zip(rows, predicted)):
            found = {source(k) for k in mids if k.startswith(('S2-', 'S3-'))}
            missing = (2, 3) if not found else tuple({2, 3} - found)
            needed.append(missing)
            for dest in missing:
                queries[dest].append((i, row, 'original'))
                # New: search with an already-matched opposite-source record.
                anchors = [self.keycache[k] for k in mids if k in self.keycache and source(k) != dest]
                if anchors:
                    best = max(anchors, key=lambda x: sum(normalize_similarity(row, x)[:2]))
                    nr, ad, ok = normalize_similarity(row, best)
                    if ok and (nr >= .75 or ad >= .86):
                        queries[dest].append((i, best, 'anchor'))
        if not queries:
            return [[] for _ in rows]
        self.activate(country, queries)
        # Per-entity new ID -> [S1-derived combined cosine or -1, anchor_support]
        proposals = [dict() for _ in rows]
        for dest, tasks in queries.items():
            seeds = [entry[1] for entry in tasks]
            for field in ('combined', 'name', 'addr'):
                vector = np.ascontiguousarray(self.enc[field].transform(seeds, batch_size=self.a.batch), np.float32)
                ix, lookup = self.indexes[(dest, field)]
                sims, labels = ix.search(vector, min(self.a.k, ix.ntotal))
                self.total_queries += len(seeds)
                for (i, seed, kind), ss, ll in zip(tasks, sims, labels):
                    for sim, label in zip(ss, ll):
                        if label < 0:
                            continue
                        cid = lookup[int(label)]
                        if cid in seen[i] or cid in predicted[i]:
                            continue
                        current = proposals[i].setdefault(cid, [-1., False])
                        if kind == 'original' and field == 'combined':
                            current[0] = max(current[0], float(sim))
                        if kind == 'anchor':
                            current[1] = True
        all_keys = [key for p in proposals for key in p]
        self.get(all_keys)
        feature_pairs, refs, stats = [], [], [[] for _ in rows]
        for i, (r, proposal, missing) in enumerate(zip(rows, proposals, needed)):
            shortlists = defaultdict(list)
            for cid, (sim, anchor) in proposal.items():
                target = self.keycache.get(cid)
                if target is None or target[3] != country:
                    continue
                n, ad, ok = normalize_similarity(r, target)
                if not ok and not (n > .985 and ad > .92):
                    continue
                if n < .53 and ad < .78:
                    continue
                priority = max(n, ad) + .35 * min(n, ad) + .08 * max(sim, 0.) + .03 * anchor
                shortlists[source(cid)].append((priority, cid, sim, anchor, n, ad, ok))
            for src in missing:
                for _, cid, sim, anchor, n, ad, ok in sorted(shortlists[src], reverse=True)[:self.a.shortlist]:
                    target = self.keycache[cid]
                    refs.append((i, cid, n, ad, ok, anchor))
                    feature_pairs.append((r, target, sim))
        if not feature_pairs:
            return stats
        feats = np.asarray([v6.extra_features(*pair) for pair in feature_pairs], dtype=np.float32)
        p = self.models[0].predict_proba(feats)[:, 1]
        if len(self.models) > 1:
            p = self.weight * p + (1. - self.weight) * self.models[1].predict_proba(feats)[:, 1]
        self.total_features += len(feats)
        for (i, cid, n, ad, ok, anchor), prob in zip(refs, p):
            stats[i].append((cid, float(prob), float(n), float(ad), bool(ok), bool(anchor)))
        return stats

    def close(self):
        self.indexes.clear()
        self.db.close()


def is_accepted(candidate, policy, threshold, france=False):
    cid, p, n, ad, ok, anchor = candidate
    if not ok:
        return False
    if france:
        # No France labels exist: require substantially stronger textual evidence.
        return p >= max(.94, threshold) and n >= .975 and ad >= .94
    if p < threshold:
        return False
    if policy == 'strict':
        return (n >= .94 and ad >= .84) or (n >= .995 and ad == 0.)
    if policy == 'balanced':
        return (n >= .84 and ad >= .70) or (n >= .98 and ad == 0.) or (ad >= .97 and n >= .74)
    if policy == 'model':
        return (n >= .70 and ad >= .55) or (n >= .97 and ad == 0.) or (ad >= .94 and n >= .68)
    raise ValueError(policy)


def opts(a):
    return {'k':a.k, 'ef':a.ef, 'shortlist':a.shortlist, 'feature_schema':'v6_extra_features', 'version':12}


def read_local(a):
    folder = a.baseline
    files = ['fresh_validation.json', 'validation_scores.json', 'validation.json']
    for file in files:
        if not (folder / file).exists():
            raise FileNotFoundError('Missing V10 local validation: ' + str(folder / file))
    data = json.loads((folder / files[0]).read_text(encoding='utf8'))
    scores = json.loads((folder / files[1]).read_text(encoding='utf8'))
    meta = json.loads((folder / files[2]).read_text(encoding='utf8'))
    threshold = float(meta['threshold'])
    rows, truth = [tuple(r) for r in data['rows']], {k:set(v) for k,v in data['truth'].items()}
    candidates = [set(scores[r[0]][0]) for r in rows]
    base = [ids(','.join(cid for cid,p in zip(*scores[r[0]]) if p >= threshold)) for r in rows]
    return rows, truth, candidates, base, threshold


def score_subset(rows, truth, baseline, extra, chosen, subset, france=False):
    policy, threshold = chosen
    vals = []
    tp = fp = fn = new = 0
    for i in subset:
        added = {x[0] for x in extra[i] if is_accepted(x, policy, threshold, france)}
        prev = set(baseline[i])
        added -= prev
        pred = prev | added
        actual = truth[rows[i][0]]
        vals.append(metric(pred, actual))
        tp += len(pred & actual)
        fp += len(pred - actual)
        fn += len(actual - pred)
        new += len(added)
    return {'macro_f05':round(float(np.mean(vals)), 7), 'tp':tp, 'fp':fp, 'fn':fn, 'new_matches':new, 'rows':len(vals)}


def validate(a):
    rows, truth, seen, baseline, original_threshold = read_local(a)
    LOG.info('Validating targeted wider ANN on %d existing labeled training examples', len(rows))
    engine = Expander(a, a.root / 'artifacts_v7_fresh')
    added = [[] for _ in rows]
    try:
        for country in sorted({r[3] for r in rows}):
            positions = [i for i,r in enumerate(rows) if r[3] == country]
            for start in range(0, len(positions), a.batch):
                indexes = positions[start:start+a.batch]
                t = time.monotonic()
                output = engine.expand([rows[i] for i in indexes], [baseline[i] for i in indexes], [seen[i] for i in indexes])
                for i, x in zip(indexes, output):
                    added[i] = x
                LOG.info('VALIDATE %s %d/%d elapsed %.1fs new candidates=%d',country,min(start+a.batch,len(positions)),len(positions),time.monotonic()-t,sum(map(len,output)))
    finally:
        engine.close()
    from numpy.random import default_rng
    order = default_rng(20260928).permutation(len(rows)).tolist()
    tuning, report = order[:len(order)//2], order[len(order)//2:]
    base_tune = score_subset(rows, truth, baseline, added, ('strict', 999.), tuning)
    base_hold = score_subset(rows, truth, baseline, added, ('strict', 999.), report)
    # Evaluate a compact, fixed grid on tuning only; report subset is diagnostic.
    thresholds = sorted(set([.65,.75,.85,.92,.97,round(original_threshold,3),round(max(original_threshold,.8),3)]))
    options = {}
    for policy in ('strict','balanced','model'):
        for th in thresholds:
            name = f'{policy}:{th:.3f}'
            options[name] = score_subset(rows, truth, baseline, added, (policy, th), tuning)
    # Prefer the most conservative threshold and gate when validation scores tie.
    ranked = sorted(options, key=lambda n: (options[n]['macro_f05'],
                    float(n.split(':')[1]), {'strict':2,'balanced':1,'model':0}[n.split(':')[0]]),
                    reverse=True)
    selected = 'off'
    evaluation = {}
    if ranked:
        candidate = ranked[0]
        policy, th = candidate.split(':')
        new_held = score_subset(rows, truth, baseline, added, (policy,float(th)), report)
        per_country = {}
        for country in sorted({r[3] for r in rows}):
            indices = [i for i in report if rows[i][3] == country]
            if indices:
                per_country[country] = {'baseline':score_subset(rows,truth,baseline,added,('strict',999.),indices),
                                        'new':score_subset(rows,truth,baseline,added,(policy,float(th)),indices)}
        evaluation = {'selected_candidate':candidate,'candidate_tune':options[candidate], 'candidate_heldout':new_held, 'country_heldout':per_country}
        # Guard against a 1,000-row tuning fluctuation or a country-specific crash.
        improves = options[candidate]['macro_f05'] >= base_tune['macro_f05'] + .0015
        corroborates = new_held['macro_f05'] >= base_hold['macro_f05']
        safe_countries = all(v['new']['macro_f05'] >= v['baseline']['macro_f05'] - .002 for v in per_country.values())
        if improves and corroborates and safe_countries and options[candidate]['new_matches']:
            selected = candidate
    retrieval = []
    for i, r in enumerate(rows):
        possible = seen[i] | {x[0] for x in added[i]}  # union ceiling; not necessarily all real-world matches
        retrieval.append(metric(possible & truth[r[0]], truth[r[0]]))
    report_obj = {'settings':opts(a),'v10_threshold':original_threshold,'selected':selected,
                  'baseline_tune':base_tune,'baseline_report':base_hold,'evaluation':evaluation,
                  'new_candidate_rows':sum(bool(x) for x in added),
                  'new_candidate_pairs':sum(map(len, added)),
                  'expanded_retrieval_oracle':float(np.mean(retrieval)),
                  'notes':'Previous local validation was inspected; confirm actual test performance on leaderboard. France has no labels.'}
    safe_json(a.output / 'v12_validation.json',report_obj)
    LOG.info('VALIDATION RESULT %s',json.dumps(report_obj,ensure_ascii=False))
    return report_obj


def validate_baseline_paths(a):
    paths = [a.baseline / 'matching_results.tsv', a.baseline / 'candidate_pairs.tsv']
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError('Need both complete V10 submission files: ' + str(paths))
    with paths[0].open('rb') as m, paths[1].open('rb') as c:
        if m.readline().strip() != TARGET_HEADERS[0] or c.readline().strip() != TARGET_HEADERS[1]:
            raise RuntimeError('V10 output TSV headers incorrect')
    return paths


def offsets(a, counts, paths):
    marker = a.output / 'v12_baseline_offsets.json'
    fingerprint = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    if marker.exists():
        data = json.loads(marker.read_text(encoding='utf8'))
        if data['inputs'] == fingerprint and data['counts'] == counts:
            return data['offsets']
    country_offsets = {}
    with paths[0].open('rb') as m, paths[1].open('rb') as c:
        m.readline();c.readline()
        for country in sorted(counts):
            country_offsets[country] = [m.tell(), c.tell()]
            for _ in range(counts[country]):
                ml,cl = m.readline(), c.readline()
                if not ml or not cl or ml.partition(b'\t')[0] != cl.partition(b'\t')[0]:
                    raise RuntimeError(f'V10 file alignment invalid while locating {country}')
        if m.readline() or c.readline():
            raise RuntimeError('V10 file length exceeds expected S1 entity count')
    safe_json(marker,{'inputs':fingerprint,'counts':counts,'offsets':country_offsets})
    return country_offsets


def copy_fallback(m, c, fm, fc, remaining):
    for _ in range(remaining):
        x,y=m.readline(),c.readline()
        if not x or not y or x.partition(b'\t')[0] != y.partition(b'\t')[0]:
            raise RuntimeError('Baseline mismatch while copying remaining predictions')
        fm.write(x);fc.write(y)


def run_test(a, report):
    selected = report['selected']
    paths = validate_baseline_paths(a)
    final = [a.output / 'matching_results.tsv', a.output / 'candidate_pairs.tsv']
    # Always secure complete baseline output before enrichment starts.
    if not all(p.exists() for p in final):
        for src, dst in zip(paths, final):
            temp = dst.with_suffix('.baseline_tmp')
            shutil.copyfile(src,temp)
            os.replace(temp,dst)
        LOG.info('Copied complete V10 fallback to %s', a.output)
    if selected == 'off':
        # Never leave an older enriched output in this folder if the new policy
        # selects OFF. Reset to the original validated baseline byte-for-byte.
        for src, dst in zip(paths, final):
            if sha256(src) != sha256(dst):
                temp = dst.with_name(dst.name + '.reset_tmp')
                shutil.copyfile(src, temp)
                os.replace(temp, dst)
        LOG.warning('No corroborated local improvement: V12 is identical to V10; DO NOT use a submission attempt.')
        safe_json(a.output/'v12_run.json',{'selected':'off','changed_rows':0,'v10_sha256':sha256(paths[0]), 'result':'BASELINE ONLY'})
        return
    policy, threshold_text = selected.split(':')
    threshold = float(threshold_text)
    work = a.root / 'artifacts_v8_test'
    counts = json.loads((work / 's1_by_country' / 'complete.json').read_text(encoding='utf8'))
    start_offsets = offsets(a, counts, paths)
    signature = {'settings':report['settings'],'selected':selected,'baseline':[[p.stat().st_size,p.stat().st_mtime_ns] for p in paths],'france':bool(a.france)}
    signature_path = a.output / 'v12_run_settings.json'
    parts = a.output / 'parts'
    parts.mkdir(exist_ok=True)
    if signature_path.exists() and json.loads(signature_path.read_text(encoding='utf8')) != signature:
        if list(parts.glob('*.checkpoint.json')) or list(parts.glob('*.done.json')):
            raise RuntimeError('V12 settings differ from existing country checkpoints. Choose a NEW --output folder.')
    safe_json(signature_path,signature)
    engine = Expander(a, work)
    began = time.monotonic()
    budget_end = began + 60*a.minutes
    changed = 0
    new_count = 0
    try:
        # Country work order favors labeled-distribution countries; final output
        # is assembled alphabetically to match the existing V10 order.
        preferred = [c for c in ('india','us','france') if c in counts]
        preferred += [c for c in counts if c not in preferred]
        for country in preferred:
            done = parts / f'{country}.done.json'
            checkpoint_file = parts / f'{country}.checkpoint.json'
            om = parts / f'{country}.matching.part'
            oc = parts / f'{country}.candidates.part'
            if done.exists():
                info = json.loads(done.read_text(encoding='utf8'))
                if info['rows'] == counts[country] and om.exists() and oc.exists() and om.stat().st_size == info['m'] and oc.stat().st_size == info['c']:
                    changed += info.get('changed_rows', 0)
                    new_count += info.get('new_matches', 0)
                    LOG.info('Skipping completed %s',country)
                    continue
                raise RuntimeError('Damaged completed V12 country files: '+country)
            ck = json.loads(checkpoint_file.read_text(encoding='utf8')) if checkpoint_file.exists() else {'rows':0,'m':0,'c':0,'changed_rows':0,'new_matches':0}
            if ck['rows'] and (not om.exists() or not oc.exists()):
                raise RuntimeError('V12 checkpoint without its partial files: '+country)
            completed, changed_country, new_country = ck['rows'],ck.get('changed_rows',0),ck.get('new_matches',0)
            with paths[0].open('rb') as m, paths[1].open('rb') as c, (work/'s1_by_country'/f'{country}.tsv').open(encoding='utf8',newline='') as fi, om.open('r+b' if om.exists() else 'w+b') as fm, oc.open('r+b' if oc.exists() else 'w+b') as fc:
                m.seek(start_offsets[country][0]);c.seek(start_offsets[country][1]);reader=csv.reader(fi,delimiter='\t')
                for _ in range(completed):
                    m.readline();c.readline();next(reader)
                fm.truncate(ck['m']);fc.truncate(ck['c']);fm.seek(0,2);fc.seek(0,2)
                while completed < counts[country]:
                    active = time.monotonic() < budget_end and (country != 'france' or a.france)
                    if not active:
                        remaining = counts[country] - completed
                        LOG.warning('Country %s: preserving remaining %d original V10 rows (budget/France restriction)',country,remaining)
                        copy_fallback(m,c,fm,fc,remaining)
                        completed += remaining
                        fm.flush();fc.flush();os.fsync(fm.fileno());os.fsync(fc.fileno())
                        safe_json(checkpoint_file,{'rows':completed,'m':fm.tell(),'c':fc.tell(),'changed_rows':changed_country,'new_matches':new_country})
                        break
                    batch = min(a.batch,counts[country]-completed)
                    rows=[];mb=[];cb=[]
                    for _ in range(batch):
                        record=tuple(next(reader))
                        ml,cl=m.readline(),c.readline()
                        if not ml or not cl:
                            raise RuntimeError('Unexpected truncated V10 files')
                        ml=ml.decode('utf8').rstrip('\r\n').split('\t',1)
                        cl=cl.decode('utf8').rstrip('\r\n').split('\t',1)
                        if len(ml)!=2 or len(cl)!=2 or ml[0]!=record[0] or cl[0]!=record[0]:
                            raise RuntimeError('V10 and S1 country partition have different row order: '+country)
                        rows.append(record);mb.append(ids(ml[1]));cb.append(ids(cl[1]))
                    started = time.monotonic()
                    proposals=engine.expand(rows,mb,[set(cids) for cids in cb])
                    for r,old,old_c,new in zip(rows,mb,cb,proposals):
                        extra=[x[0] for x in new if is_accepted(x,policy,threshold,france=country=='france') and x[0] not in old]
                        if extra:
                            changed_country += 1
                            new_country += len(extra)
                        predicted=list(dict.fromkeys(old+extra))
                        candidates=list(dict.fromkeys(old_c+[x[0] for x in new]+predicted))
                        fm.write((r[0]+'\t'+','.join(predicted)+'\n').encode('utf8'))
                        fc.write((r[0]+'\t'+','.join(candidates)+'\n').encode('utf8'))
                    completed+=batch
                    fm.flush();fc.flush();os.fsync(fm.fileno());os.fsync(fc.fileno())
                    safe_json(checkpoint_file,{'rows':completed,'m':fm.tell(),'c':fc.tell(),'changed_rows':changed_country,'new_matches':new_country})
                    LOG.info('TEST %s %d/%d changed=%d extras=%d rate=%.1f entities/s',country,completed,counts[country],changed_country,new_country,batch/max(.001,time.monotonic()-started))
                safe_json(done,{'rows':completed,'m':fm.tell(),'c':fc.tell(),'changed_rows':changed_country,'new_matches':new_country})
            changed+=changed_country
            new_count+=new_country
            LOG.info('Country %s COMPLETE. Changed rows=%d',country,changed_country)
    finally:
        engine.close()
    # Full-file assembly is atomic. Existing complete baseline survives interruptions.
    for idx,(final_file,suffix,header) in enumerate(zip(final,('matching','candidates'),TARGET_HEADERS)):
        temp = final_file.with_name(final_file.name+'.building')
        with temp.open('wb') as dst:
            dst.write(header+b'\n')
            for country in sorted(counts):
                with (parts/f'{country}.{suffix}.part').open('rb') as src:
                    shutil.copyfileobj(src,dst,1024*1024)
        os.replace(temp,final_file)
    actual_sha=sha256(final[0]);original_sha=sha256(paths[0])
    result={'selected':selected,'total_rows':sum(counts.values()),'changed_rows':changed,'additional_predictions':new_count,
            'v10_sha256':original_sha,'v12_sha256':actual_sha,'different_from_v10':actual_sha!=original_sha,
            'elapsed_minutes':round((time.monotonic()-began)/60,2),'france_enrichment':a.france}
    safe_json(a.output/'v12_run.json',result)
    LOG.info('SUBMISSION READY %s',json.dumps(result))
    if actual_sha == original_sha:
        LOG.warning('V12 matches V10 byte-for-byte. DO NOT spend a submission on it.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['all','validate','test'],default='all')
    parser.add_argument('--root',type=Path,default=Path('student_resource'))
    parser.add_argument('--baseline',type=Path,default=Path('student_resource/output_v10'))
    parser.add_argument('--output',type=Path,default=Path('student_resource/output_v12'))
    parser.add_argument('--minutes',type=float,default=75.,help='Max active ANN enrichment minutes; always completes output with V10 fallback')
    parser.add_argument('--batch',type=int,default=160)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--k',type=int,default=16,help='Neighbors PER source, channel, seed (V10 used 4)')
    parser.add_argument('--ef',type=int,default=80)
    parser.add_argument('--shortlist',type=int,default=8,help='Max new candidates scored per missing source per entity')
    parser.add_argument('--cuda',action='store_true',help='Use existing CUDA PyTorch for encoder projection; CPU default avoids GPU overhead')
    parser.add_argument('--france',action='store_true',help='Experimental conservative France enrichment despite NO France training labels')
    a=parser.parse_args()
    if min(a.minutes,a.batch,a.k,a.ef,a.shortlist,a.threads)<=0:
        parser.error('All numeric arguments must be positive')
    if a.output.resolve()==a.baseline.resolve():
        parser.error('--output must differ from --baseline')
    a.output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',stream=sys.stdout)
    faiss.omp_set_num_threads(a.threads)
    torch.set_num_threads(min(a.threads,8))
    # Keep repeated per-string transformations bounded across up to 2k validation rows.
    v4.name_key=lru_cache(maxsize=40000)(v4.name_key)
    v4.addr_key=lru_cache(maxsize=40000)(v4.addr_key)
    for fname in ('clean_alnum','initials','digit_tokens','char_ngrams','ocr_fold'):
        setattr(v6,fname,lru_cache(maxsize=40000)(getattr(v6,fname)))
    validate_baseline_paths(a)
    path=a.output/'v12_validation.json'
    if a.mode in ('validate','all'):
        report=validate(a)
    else:
        if not path.exists():
            raise FileNotFoundError('Run --mode validate first to select a locally tested recovery strategy')
        report=json.loads(path.read_text(encoding='utf8'))
        if report['settings']!=opts(a):
            raise RuntimeError('Runtime retrieval settings differ from V12 validation. Keep --k/--ef/--shortlist unchanged.')
    if a.mode in ('test','all'):
        run_test(a,report)
        LOG.info('Run the official validator. Only the real leaderboard measures hidden-test accuracy.')


if __name__=='__main__':
    main()
