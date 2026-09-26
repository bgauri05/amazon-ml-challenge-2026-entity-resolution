#!/usr/bin/env python3
"""Deadline-safe V9 inference: existing V8 model + indexed exact SQL, optional bounded ANN.

A valid V1 matching/candidate pair is copied to the output FIRST as a crash-safe
fallback. V9 then improves country-partitioned rows and atomically replaces these
files on successful completion. A time budget bounds *active* V9 inference; rows
not enhanced within the budget retain the existing V1 predictions. No training,
external data or rebuilding of target indexes. No leaderboard score guarantee.
"""
from __future__ import annotations
import __main__, argparse, csv, gc, json, logging, os, re, shutil, sqlite3, time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import numpy as np
import faiss
import joblib
import torch
import xgboost as xgb
from rapidfuzz import fuzz
import model_v4_hybrid as v4
import model_v6_fast as v6
import model_v7_fresh as v7

LOG = logging.getLogger('v9_safe')
MHEAD = ['source1_entity_id', 'matched_entity_ids']
CHEAD = ['source1_entity_id', 'candidate_entity_ids']


def load_baseline(path):
    if not path.is_file():
        raise FileNotFoundError(f'Full-test V1 fallback missing: {path}')
    result = {}
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if reader.fieldnames != MHEAD:
            raise ValueError(f'Unexpected V1 header: {reader.fieldnames}')
        for row in reader:
            rid = row['source1_entity_id']
            if rid in result:
                raise ValueError(f'Duplicate S1 in V1: {rid}')
            result[rid] = row['matched_entity_ids']
    LOG.info('Loaded %d baseline V1 predictions', len(result))
    return result


def atomic_copy(src, dst):
    tmp = dst.with_name(dst.name + '.initial_copy')
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def seed_fallback(output, v1, v1c, baseline, counts):
    """Produce complete fallback files BEFORE beginning expensive inference."""
    expected = sum(counts.values())
    if len(baseline) != expected:
        raise ValueError(f'V1 has {len(baseline)} entities; test expects {expected}. '
                         'Do not use an incomplete or different-test V1 file.')
    output.mkdir(parents=True, exist_ok=True)
    if not (output / 'matching_results.tsv').is_file():
        atomic_copy(v1, output / 'matching_results.tsv')
    if not (output / 'candidate_pairs.tsv').is_file():
        if v1c.is_file():
            # Use the genuine candidate file from the completed V1 pipeline.
            atomic_copy(v1c, output / 'candidate_pairs.tsv')
        else:
            temp = output / 'candidate_pairs.initial_copy'
            with temp.open('w', encoding='utf-8', newline='') as f:
                wr = csv.writer(f, delimiter='\t', lineterminator='\n')
                wr.writerow(CHEAD)
                for rid, mids in baseline.items():
                    wr.writerow((rid, mids))
            os.replace(temp, output / 'candidate_pairs.tsv')
    LOG.info('Complete initial fallback files exist in %s', output)


def exact_batch(db, rows, cap):
    """Country-batched index probes by exact canonical name and exact address."""
    proposals = [{} for _ in rows]
    for col, keyfun, other, minlen in (
        ('name_key', v4.name_key, v4.addr_key, 3),
        ('addr_key', v4.addr_key, v4.name_key, 7),
    ):
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            key = keyfun(r[1] if col == 'name_key' else r[2])
            if len(key) >= minlen:
                groups[(r[3], key)].append(i)
        bycountry = defaultdict(list)
        for country, key in groups:
            bycountry[country].append(key)
        for country, keys in bycountry.items():
            for start in range(0, len(keys), 150):
                section = keys[start:start+150]
                marks = ','.join('?' * len(section))
                sql = (f'SELECT id,name,addr,{col} FROM targets '
                       f'WHERE country=? AND {col} IN ({marks})')
                matches = defaultdict(list)
                for cid, name, addr, key in db.execute(sql, [country, *section]):
                    if len(matches[key]) < min(cap * 8, 160):
                        matches[key].append((cid, name, addr))
                for key in section:
                    hits = matches.get(key, ())
                    for i in groups[(country, key)]:
                        if len(hits) > cap:
                            probe = other(rows[i][2] if col == 'name_key' else rows[i][1])
                            selected = sorted(
                                hits,
                                key=lambda x: fuzz.WRatio(probe, other(x[2] if col == 'name_key' else x[1])),
                                reverse=True,
                            )[:cap]
                        else:
                            selected = hits
                        for cid, name, addr in selected:
                            proposals[i][cid] = (name, addr)
    return proposals


def model_bundle(models_dir, threads):
    meta_path = models_dir / 'v8_metadata.json'
    if not meta_path.is_file():
        raise FileNotFoundError(f'V8 trained model metadata missing: {meta_path}')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if meta['feature_count'] != v6.FEATURE_COUNT:
        raise RuntimeError('V8 and local V6 feature counts differ. Do not score.')
    chosen = meta['selected']
    threshold = float(meta['selection']['threshold'])
    if chosen == 'v7_original':
        paths, weight = [models_dir/'v8_selected_v7.json'], 1.
    elif chosen in ('precision', 'recall'):
        paths, weight = [models_dir/f'v8_{chosen}.json'], 1.
    elif chosen.startswith('blend_'):
        paths = [models_dir/'v8_precision.json', models_dir/'v8_recall.json']
        weight = int(chosen.split('_')[1])/100.
    else:
        raise RuntimeError(f'Unknown selected V8 model: {chosen}')
    loaded = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Missing V8 model file: {path}')
        m = xgb.XGBClassifier()
        m.load_model(str(path))
        m.set_params(device='cpu', n_jobs=threads)
        loaded.append(m)
    LOG.info('Using existing V8 %s, threshold %.3f', chosen, threshold)
    return loaded, weight, threshold


def ann_for_country(work, country, ef):
    out = []
    for source in (2, 3):
        prefix = f'{source}_{country}'
        ix_path = work/'combined_indexes'/f'{prefix}.faiss'
        ids_path = work/'combined_indexes'/f'{prefix}.ids.joblib'
        if not ix_path.is_file() or not ids_path.is_file():
            continue
        idx = faiss.read_index(str(ix_path))
        idx.hnsw.efSearch = ef
        ids = joblib.load(ids_path)
        if idx.ntotal != len(ids):
            raise RuntimeError(f'ANN index-to-ID misalignment: {prefix}')
        out.append((idx, ids))
    return out


def numbers_compatible(a, b):
    x, y = set(re.findall(r'\d+', a)), set(re.findall(r'\d+', b))
    return not (x and y and x.isdisjoint(y))


def strict_hits(row, exact, min_name, min_addr):
    """Force only strong two-field agreements; reject contradictory address numbers."""
    nk, ak = v4.name_key(row[1]), v4.addr_key(row[2])
    accepted = []
    if not nk or len(ak) < 7:
        return accepted
    for cid, (name, addr) in exact.items():
        bn, ba = v4.name_key(name), v4.addr_key(addr)
        if len(ba) < 7 or not numbers_compatible(ak, ba):
            continue
        if (nk == bn and ak == ba) or (
            fuzz.ratio(nk, bn)/100. >= min_name
            and fuzz.ratio(ak, ba)/100. >= min_addr
        ):
            accepted.append(cid)
    return accepted


def score_batch(rows, proposals, exacts, db, models, weight, threshold, max_pairs):
    # Reuse SQL metadata from exact blocking; fetch only additional ANN IDs.
    targets = {}
    for row, exact in zip(rows, exacts):
        for cid, (name, addr) in exact.items():
            targets[cid] = (cid, name, addr, row[3])
    missing = list(dict.fromkeys(cid for p in proposals for cid, _ in p if cid not in targets))
    for start in range(0, len(missing), 350):
        ids = missing[start:start+350]
        marks = ','.join('?' * len(ids))
        for rid, name, addr, country in db.execute(
            f'SELECT id,name,addr,country FROM targets WHERE id IN ({marks})', ids
        ):
            targets[rid] = (rid, name, addr, country)
    feats, flat, spans = [], [], []
    for row, pairs in zip(rows, proposals):
        first = len(flat)
        for cid, sim in pairs[:max_pairs]:
            rec = targets.get(cid)
            if rec is not None:
                feats.append(v6.extra_features(row, rec, sim))
                flat.append(cid)
        spans.append((first, len(flat)))
    if feats:
        matrix = np.asarray(feats, dtype=np.float32)
        probs = models[0].predict_proba(matrix)[:, 1]
        if len(models) > 1:
            probs = weight*probs + (1.-weight)*models[1].predict_proba(matrix)[:, 1]
    else:
        probs = np.empty(0, dtype=np.float32)
    return [
        [rid for rid, prob in zip(flat[lo:hi], probs[lo:hi]) if prob >= threshold]
        for lo, hi in spans
    ]


def save_checkpoint(path, rows, matching_partial, candidate_partial):
    # Flush/fsync both data files BEFORE atomically committing their byte offsets.
    offsets = {'rows': rows,
               'matching_bytes': matching_partial.stat().st_size,
               'candidate_bytes': candidate_partial.stat().st_size}
    tmp = path.with_name(path.name + '.new')
    tmp.write_text(json.dumps(offsets), encoding='utf-8')
    os.replace(tmp, path)


def check_completed(path, expected):
    if not path.is_file():
        return False
    with path.open(encoding='utf-8') as f:
        return sum(1 for _ in f) == expected


def main():
    p = argparse.ArgumentParser(description='V9: bounded, resumable V8 exact-blocking inference plus complete V1 fallback')
    p.add_argument('--data', type=Path, default=Path('student_resource/dataset'))
    p.add_argument('--v1', type=Path, default=Path('v1_backup/matching_results.tsv'))
    p.add_argument('--v1-candidates', type=Path, default=Path('v1_backup/candidate_pairs.tsv'))
    p.add_argument('--output', type=Path, default=Path('student_resource/output_v9'))
    p.add_argument('--minutes', type=float, default=200., help='Maximum active V9 computation time; remaining rows use V1')
    p.add_argument('--batch', type=int, default=400)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--exact-cap', type=int, default=10)
    p.add_argument('--max-pairs', type=int, default=24)
    p.add_argument('--min-name', type=float, default=.93)
    p.add_argument('--min-addr', type=float, default=.91)
    p.add_argument('--ann', action='store_true', help='Optional combined ANN; do NOT enable without speed benchmark')
    p.add_argument('--ann-k', type=int, default=8)
    p.add_argument('--ef', type=int, default=40)
    p.add_argument('--country-order', default='india,us,france', help='Comma-separated priority; known training countries first by default')
    p.add_argument('--benchmark-rows', type=int, default=0, help='Process only N rows, print timings, and exit without replacing final files')
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if a.batch <= 0 or a.threads <= 0 or a.minutes < 0 or a.exact_cap <= 0:
        p.error('batch, threads, exact-cap must be positive and minutes nonnegative')
    v4.name_key = lru_cache(maxsize=50000)(v4.name_key)
    v4.addr_key = lru_cache(maxsize=50000)(v4.addr_key)
    faiss.omp_set_num_threads(a.threads)
    torch.set_num_threads(a.threads)
    root = a.data.parent
    work = root/'artifacts_v8_test'
    models_dir = root/'models_v8'
    partition = work/'s1_by_country'
    db_path = work/'train_targets.sqlite'
    partition_info = partition/'complete.json'
    if not db_path.is_file() or not partition_info.is_file():
        raise FileNotFoundError('Completed V8 test database or country partitions missing. Do not rebuild; verify artifact paths.')
    counts = json.loads(partition_info.read_text(encoding='utf-8'))
    baseline = load_baseline(a.v1)
    if len(baseline) != sum(counts.values()):
        raise ValueError('V1 backup is not complete for this test set')
    if not a.benchmark_rows:
        seed_fallback(a.output, a.v1, a.v1_candidates, baseline, counts)
    models, weight, threshold = model_bundle(models_dir, a.threads)
    db = sqlite3.connect(f'file:{db_path.resolve().as_posix()}?mode=ro', uri=True, timeout=10)
    db.execute('PRAGMA query_only=ON')
    db.execute('PRAGMA cache_size=-65536')
    db.execute('PRAGMA mmap_size=268435456')
    encoder = None
    if a.ann:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        __main__.FieldEncoder = v7.FieldEncoder
        encoder = v7.load_encoder(root/'artifacts_v7_fresh'/'encoder_combined_d64_s60000.joblib', device)
    progress = a.output/'progress'
    progress.mkdir(parents=True, exist_ok=True)
    total = sum(counts.values())
    complete, improved, fallback = 0, 0, 0
    started = time.monotonic()
    budget = a.minutes*60.
    preferred = [c.strip() for c in a.country_order.split(',') if c.strip() in counts]
    country_order = list(dict.fromkeys(preferred + sorted(counts)))
    LOG.info('Inference country priority: %s', country_order)
    for country in country_order:
        src = partition/f'{country}.tsv'
        final_m = progress/f'{country}.tsv'
        final_c = progress/f'{country}.candidates.tsv'
        part_m = progress/f'{country}.partial'
        part_c = progress/f'{country}.candidates.partial'
        checkpoint = progress/f'{country}.checkpoint.json'
        if not a.benchmark_rows and check_completed(final_m, counts[country]) and check_completed(final_c, counts[country]):
            LOG.info('Already complete: %s', country)
            complete += counts[country]
            continue
        resume = 0
        if not a.benchmark_rows and checkpoint.is_file() and part_m.is_file() and part_c.is_file():
            saved = json.loads(checkpoint.read_text(encoding='utf-8'))
            msize, csize = saved['matching_bytes'], saved['candidate_bytes']
            if msize <= part_m.stat().st_size and csize <= part_c.stat().st_size and 0 <= saved['rows'] <= counts[country]:
                with part_m.open('rb+') as f:
                    f.truncate(msize)
                with part_c.open('rb+') as f:
                    f.truncate(csize)
                resume = saved['rows']
                LOG.info('Resuming %s at %d/%d', country, resume, counts[country])
        if resume == 0:
            checkpoint.unlink(missing_ok=True)
        complete += resume
        country_done = resume
        ann = []
        if encoder is not None and time.monotonic()-started < budget:
            LOG.info('Reading existing combined ANN for %s', country)
            ann = ann_for_country(work, country, a.ef)
        with src.open(encoding='utf-8', newline='') as fi, \
             part_m.open('a' if resume else 'w', encoding='utf-8', newline='') as fm, \
             part_c.open('a' if resume else 'w', encoding='utf-8', newline='') as fc:
            reader = csv.reader(fi, delimiter='\t')
            for _ in range(resume):
                next(reader)
            wm = csv.writer(fm, delimiter='\t', lineterminator='\n')
            wc = csv.writer(fc, delimiter='\t', lineterminator='\n')
            while True:
                raw = []
                for _ in range(a.batch):
                    try:
                        raw.append(next(reader))
                    except StopIteration:
                        break
                if not raw:
                    break
                rows = [tuple(r) for r in raw]
                if a.benchmark_rows and complete + len(rows) > a.benchmark_rows:
                    rows = rows[:a.benchmark_rows-complete]
                    if not rows:
                        break
                timed_out = time.monotonic()-started >= budget
                if timed_out:
                    for r in rows:
                        prediction = baseline.get(r[0], '')
                        wm.writerow((r[0], prediction))
                        wc.writerow((r[0], prediction))
                    fallback += len(rows)
                else:
                    t0 = time.monotonic()
                    exacts = exact_batch(db, rows, a.exact_cap)
                    t_exact = time.monotonic()
                    proposals = [[(rid, -1.) for rid in hits] for hits in exacts]
                    if ann:
                        vectors = encoder.transform(rows, batch_size=a.batch)
                        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
                        for ix, ids in ann:
                            sims, labels = ix.search(vectors, min(a.ann_k, ix.ntotal))
                            for i, (labs, scores) in enumerate(zip(labels, sims)):
                                used = {cid for cid, _ in proposals[i]}
                                for label, sim in zip(labs, scores):
                                    if label >= 0:
                                        cid = ids[int(label)]
                                        if cid not in used:
                                            proposals[i].append((cid, float(sim)))
                                            used.add(cid)
                    t_ann = time.monotonic()
                    predictions = score_batch(rows, proposals, exacts, db, models, weight, threshold, a.max_pairs)
                    t_score = time.monotonic()
                    for r, exact, pairs, pred in zip(rows, exacts, proposals, predictions):
                        strict = strict_hits(r, exact, a.min_name, a.min_addr)
                        found = list(dict.fromkeys(pred + strict))
                        if not found:
                            found = list(filter(None, baseline.get(r[0], '').split(',')))
                        else:
                            improved += 1
                        # Include all produced candidates AND fallback final choices.
                        candidates = list(dict.fromkeys([cid for cid, _ in pairs] + found))
                        wm.writerow((r[0], ','.join(found)))
                        wc.writerow((r[0], ','.join(candidates)))
                    elapsed = time.monotonic()-t0
                    LOG.info('BATCH %s %d records: exact=%.2fs ann=%.2fs model=%.2fs total=%.2fs %.1f rows/s; time_left=%.1fmin',
                             country, len(rows), t_exact-t0, t_ann-t_exact, t_score-t_ann, elapsed,
                             len(rows)/max(elapsed, .001), max(0., (budget-(time.monotonic()-started))/60.))
                complete += len(rows)
                country_done += len(rows)
                fm.flush(); fc.flush(); os.fsync(fm.fileno()); os.fsync(fc.fileno())
                if not a.benchmark_rows:
                    save_checkpoint(checkpoint, country_done, part_m, part_c)
                if complete % 5000 < a.batch:
                    LOG.info('PROGRESS %d/%d enhanced_rows=%d fallback_rows=%d', complete, total, improved, fallback)
                if a.benchmark_rows and complete >= a.benchmark_rows:
                    LOG.info('BENCHMARK COMPLETE: %d rows in %.1fs; %.1f rows/s (includes startup); confident_rows=%d',
                             complete, time.monotonic()-started, complete/max(time.monotonic()-started,.001), improved)
                    return
        # Normal output: finalize each country atomically and retain already completed countries for resumes.
        if not a.benchmark_rows:
            if not check_completed(part_m, counts[country]) or not check_completed(part_c, counts[country]):
                raise RuntimeError(f'Country {country} incomplete: see progress for details')
            os.replace(part_m, final_m)
            os.replace(part_c, final_c)
            checkpoint.unlink(missing_ok=True)
            LOG.info('Completed country %s', country)
        del ann
        gc.collect()
    # Build complete outputs; initial V1 remains available until atomic replacements.
    for filename, header, suffix in [('matching_results.tsv', MHEAD, '.tsv'),
                                      ('candidate_pairs.tsv', CHEAD, '.candidates.tsv')]:
        dest = a.output/filename
        temp = dest.with_name(filename + '.building')
        with temp.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
            writer.writerow(header)
            for country in sorted(counts):
                with (progress/f'{country}{suffix}').open('r', encoding='utf-8') as source:
                    shutil.copyfileobj(source, handle)
        os.replace(temp, dest)
    LOG.info('SUBMISSION READY: %s (processed=%d, improvements=%d, V1 fallback=%d)',
             a.output, complete, improved, fallback)
    db.close()


if __name__ == '__main__':
    main()
