#!/usr/bin/env python3
r"""Model V6: targeted hard-positive learning and missing-address-aware matching over cached V5 candidates.

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
from collections import Counter, defaultdict
from pathlib import Path

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

LOG = logging.getLogger('model_v6')
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


def v5_features(a, b, similarity):
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


# V5 has numerous similarity features, but does not explicitly identify several
# high-value error modes observed in its validation AUDIT. No validation rows,
# names, labels or probabilities are embedded here.
def digit_tokens(s):
    return [x.lstrip('0') or '0' for x in re.findall(r'\d+', s)]


def char_ngrams(s, n=3):
    s = re.sub(r'\s+', '', s)
    return {s[i:i+n] for i in range(max(0, len(s)-n+1))}


def qgram_dice(a, b):
    x, y = char_ngrams(a), char_ngrams(b)
    return 2.*len(x & y)/(len(x)+len(y)) if x and y else 0.


def ocr_fold(s):
    # Auxiliary evidence ONLY: do not replace original fields with this lossy key.
    return s.translate(str.maketrans({'0':'o','1':'l','5':'s','8':'b'}))


def extra_features(a, b, similarity):
    f = v5_features(a, b, similarity)
    na, nb = v4.name_key(a[1]), v4.name_key(b[1])
    aa, ab = v4.addr_key(a[2]), v4.addr_key(b[2])
    ca, cb = clean_alnum(na), clean_alnum(nb)
    x, y = digit_tokens(aa), digit_tokens(ab)
    xa, xb = set(x), set(y)
    an, bn = set(na.split()) - NAME_STOP, set(nb.split()) - NAME_STOP
    n_ratio = fuzz.WRatio(na,nb)/100 if na and nb else 0.
    addr_ratio = fuzz.WRatio(aa,ab)/100 if aa and ab else 0.
    same_addr_nums = bool(xa and xb and xa == xb)
    addr_missing = not aa or not ab
    ocr_a, ocr_b = ocr_fold(ca), ocr_fold(cb)
    # These features distinguish missing evidence from contradictory evidence.
    f.extend([
        float(similarity == -1.), # SQL/FTS candidate: -1 means unknown cosine, not mismatch
        float(bool(aa and ab)),
        float(addr_missing),
        float(bool(na and nb) and ca == cb),
        float(bool(na and nb) and na == nb and addr_missing),
        float(n_ratio >= .95 and addr_missing),
        float(n_ratio >= .85 and addr_missing),
        float(n_ratio >= .85 and bool(aa and ab) and addr_ratio < .50),
        float(addr_ratio >= .95 and n_ratio >= .65),
        float(n_ratio >= .90 and same_addr_nums),
        float(bool(xa and xb) and xa.isdisjoint(xb)),
        float(bool(xa and xb) and len(xa ^ xb) == 1),
        float(bool(xa and xb) and len(xa & xb) >= 2),
        float(x[0] == y[0]) if x and y else 0.,
        float(x[-1] == y[-1]) if x and y else 0.,
        qgram_dice(ca, cb),
        qgram_dice(clean_alnum(aa), clean_alnum(ab)),
        fuzz.ratio(ocr_a, ocr_b)/100 if ocr_a and ocr_b else 0.,
        float(bool(ocr_a and ocr_b) and ocr_a == ocr_b),
        len(an & bn)/max(1,min(len(an),len(bn))) if an and bn else 0.,
        abs(len(an)-len(bn)),
        float(b[0].startswith('S2-')),
        float(b[0].startswith('S3-')),
    ])
    return f


def positive_sample_weights(X, y, base_weight=1.0):
    # New features have fixed positions from append above. Weights are chosen
    # WITHOUT looking at the validation audit or validation labels.
    weights = np.ones(len(y), dtype=np.float32)
    start = len(v5_features(('x','a','b','us'),('S2-x','a','b','us'),0.))
    pos = y == 1
    weights[pos] = base_weight
    # Give retrieved correct matches with absent addresses more representation.
    weights[pos & (X[:,start+2] > .5)] *= 1.35
    # Important for corrupted/abbreviated names and differing street spellings.
    n_sim = X[:,start+15]
    weights[pos & (n_sim < .60)] *= 1.30
    return np.minimum(weights,4.).astype(np.float32)


def audit_rescue_hints(scored, records, store, ceiling=0.7):
    """Potential missing-address rescues, derived from input fields ONLY.

    Classification thresholds must be tuned on tuning refs, never report refs.
    """
    record_by_id = {r[0]:r for r in records}
    to_check = [(rid,cid) for rid,pairs in scored.items()
                for cid,p in pairs if .05 <= p < ceiling]
    hints = set()
    for st in range(0,len(to_check),10000):
        batch=to_check[st:st+10000]
        targets=store.get_many([cid for _,cid in batch])
        for rid,cid in batch:
            left,right=record_by_id[rid],targets.get(cid)
            if right is None or (left[2] and right[2]):
                continue
            # Exact or near-exact business names with missing address, but
            # neither empty names nor generic single-token names are rescued.
            na,nb=v4.name_key(left[1]),v4.name_key(right[1])
            if not na or not nb or len(clean_alnum(na)) < 9:
                continue
            ratio=fuzz.WRatio(na,nb)
            if ratio>=94 and (len(na.split())>=2 or na==nb):
                hints.add((rid,cid))
    LOG.info('Missing-address rescue: %d eligible scored pairs',len(hints))
    return hints


def evaluate_with_rescue(scored, truth, threshold, rescue_ids, rescue_threshold):
    vals=[];tp=fp=fn=0
    for rid,pairs in scored.items():
        pred={cid for cid,p in pairs if p>=threshold or
              ((rid,cid) in rescue_ids and p>=rescue_threshold)}
        actual=truth.get(rid,set())
        vals.append(v4.entity_f05(actual,pred))
        tp+=len(pred&actual);fp+=len(pred-actual);fn+=len(actual-pred)
    return float(np.mean(vals)),(tp,fp,fn)


def tuning_rescue(scored, truth, threshold, hints):
    best=(evaluate(scored,truth,threshold)[0],None)
    for cutoff in (.10,.20,.30,.40,.50):
        val=evaluate_with_rescue(scored,truth,threshold,hints,cutoff)[0]
        if val>best[0]+0.0001:
            best=(val,cutoff)
    return best


class ExpandedStore(BASE_STORE):
    """Reuse cached V4 full-corpus indexes and add independent alternate FTS probes."""
    extras_per_probe = 75
    extra_cap = 750
    extra_enabled = True

    @staticmethod
    def probes(r):
        name = [t for t in v4.name_key(r[1]).split()
                if len(t) >= 3 and t not in NAME_STOP]
        addr = [t for t in v4.addr_key(r[2]).split()
                if len(t) >= 3 and t not in ADDR_STOP]
        nums = [s for s in re.findall(r'\d+', r[2]) if len(s) >= 2]
        p = []
        # V4 mainly probes the two longest tokens. Test *different* combinations.
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
        # Distinct queries only. Two-stage name/address retrieval is deliberate.
        seen = set()
        for field, t1, t2 in p:
            if t1 == t2:
                continue
            key = (field, tuple(sorted((t1, t2))))
            if key not in seen:
                seen.add(key)
                yield key

    def alternate_fts(self, records):
        results = [[] for _ in records]
        cache = {}
        stmt = ('SELECT t.id,t.name,t.addr FROM targets_fts '
                'JOIN targets t ON t.rowid=targets_fts.rowid '
                'WHERE targets_fts MATCH ? AND t.country=? '
                'ORDER BY bm25(targets_fts) LIMIT ?')
        for i, r in enumerate(records):
            name_hits, addr_hits = [], []
            probes = list(self.probes(r))[:7]
            for field, tokens in probes:
                q = ' AND '.join(f'{field}:"{t}"' for t in tokens)
                key = (q, r[3])
                if key not in cache:
                    cache[key] = self.conn.execute(stmt, (q, r[3], self.extras_per_probe*2)).fetchall()
                hits = cache[key]
                if field == 'name':
                    # On ambiguous name tokens, use opposite (address) evidence.
                    hits = sorted(hits, key=lambda x: (fuzz.WRatio(v4.addr_key(r[2]),v4.addr_key(x[2])),
                                                             fuzz.WRatio(v4.name_key(r[1]),v4.name_key(x[1]))),
                                  reverse=True)[:self.extras_per_probe]
                    name_hits.extend(x[0] for x in hits)
                else:
                    hits = sorted(hits, key=lambda x: (fuzz.WRatio(v4.name_key(r[1]),v4.name_key(x[1])),
                                                             fuzz.WRatio(v4.addr_key(r[2]),v4.addr_key(x[2]))),
                                  reverse=True)[:self.extras_per_probe]
                    addr_hits.extend(x[0] for x in hits)
            # Round-robin so one field cannot monopolize the candidate budget.
            for j in range(max(len(name_hits),len(addr_hits))):
                if j < len(name_hits):
                    results[i].append(name_hits[j])
                if j < len(addr_hits):
                    results[i].append(addr_hits[j])
            if len(cache) > 10000:
                cache.clear()
        return results

    def retrieve(self, records, encoder, top_k):
        base = super().retrieve(records, encoder, top_k)
        if not self.extra_enabled:
            return base
        new = self.alternate_fts(records)
        for i, row in enumerate(base):
            known = {cid for cid, _ in row}
            for cid in new[i]:
                if len(row) >= self.extra_cap:
                    break
                if cid not in known:
                    row.append((cid,-1.0))
                    known.add(cid)
        return base


def fit_xgb(X, y, device, n_estimators, threads, sample_weight=None, depth=8, child_weight=2):
    model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=depth,
        learning_rate=.045, min_child_weight=child_weight, reg_lambda=5, subsample=.85,
        colsample_bytree=.90, tree_method='hist', device=device,
        eval_metric='logloss', random_state=SEED, n_jobs=threads)
    model.fit(X, y, sample_weight=sample_weight, verbose=False)
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
        LOG.info('Hard-negative mining %d / %d S1; %d extra negatives',
                 min(total,start+len(rr)), total,len(yy))
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



def final_audit(scored, records, candidates, truth, store, threshold, hints, cutoff, out):
    """Actual selected V6 rule (classifier +/- rescue), not V5's old rule."""
    counts=Counter()
    meta={r[0]:r for r in records}
    retrieved={r[0]:{cid for cid,_ in row} for r,row in zip(records,candidates)}
    errors=[]
    for rid,pairs in scored.items():
        prob=dict(pairs)
        got={cid for cid,p in pairs if p>=threshold or
             (cutoff is not None and (rid,cid) in hints and p>=cutoff)}
        want=truth.get(rid,set())
        for cid in want-got:
            kind='retrieval_miss' if cid not in retrieved[rid] else 'classifier_false_negative'
            counts[kind]+=1
            errors.append((rid,cid,kind,prob.get(cid,-1.)))
        for cid in got-want:
            counts['classifier_false_positive']+=1
            errors.append((rid,cid,'classifier_false_positive',prob.get(cid,-1.)))
    f=out/'v6_error_audit.tsv'
    with f.open('w',encoding='utf8',newline='') as fh:
        w=csv.writer(fh,delimiter='\t',lineterminator='\n')
        w.writerow(['s1_id','s1_name','s1_address','target_id','target_name',
                    'target_address','error','predicted_probability'])
        for st in range(0,len(errors),5000):
            batch=errors[st:st+5000]
            targets=store.get_many([cid for _,cid,_,_ in batch])
            for rid,cid,kind,p in batch:
                left,right=meta[rid],targets.get(cid)
                w.writerow([rid,left[1],left[2],cid,right[1] if right else '',
                            right[2] if right else '',kind,round(p,5)])
    LOG.info('V6 actual selected-rule error audit %s',dict(counts))
    return counts


def main():
    parser=argparse.ArgumentParser(description='Experimental V6 using V5 candidate caches and cached V4 indexes')
    parser.add_argument('--data',type=Path,default=Path('student_resource/dataset'))
    parser.add_argument('--sample',type=int,default=10000)
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
    parser.add_argument('--full-test',action='store_true', help='Not supported in V6 until inference speed/schema are verified')
    parser.add_argument('--no-rescue',action='store_true',help='Disable missing-address rescue postprocessor')
    parser.add_argument('--second-model',action='store_true',help='Train alternate-depth mined XGBoost and test averaging on tuning only')
    parser.add_argument('--positive-weight',type=float,default=1.35)
    parser.add_argument('--candidate-cache',action='store_true',
                        help='Save/load candidate lists for this exact configuration/sample')
    args=parser.parse_args()
    args.candidate_cache=True
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    faiss.omp_set_num_threads(args.cpu_threads)
    torch.set_num_threads(args.cpu_threads)
    # V4 actually wrote its caches to artifacts_v3, NOT artifacts_v4.
    work=args.data.parent/'artifacts_v3'
    out=args.data.parent/'output_v6'
    model_dir=args.data.parent/'models_v6'
    out.mkdir(parents=True,exist_ok=True)
    model_dir.mkdir(parents=True,exist_ok=True)
    if args.full_test: parser.error('Full-test inference is disabled in this experimental V6; evaluate before exporting.')
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
    sampled=v4.sample_records(v4.source_path(args.data,'train',1),args.sample)
    train,valid=train_test_split(sampled,test_size=.2,random_state=SEED)
    truth=v4.read_truth(args.data/'train'/'train_ground_truth.tsv')
    key=f's{args.sample}_k{args.top_k}_m{args.max_candidates}_e{args.extra_cap}_p{args.extra_probe_results}_x{int(ExpandedStore.extra_enabled)}'
    def load_candidates(split_name,rr):
        p=args.data.parent/'output_v5'/f'{split_name}_candidates_{key}.joblib'
        if args.candidate_cache and p.exists():
            LOG.info('Loading %s V5 cached candidates: %s',split_name,p)
            cached=joblib.load(p)
            if len(cached)!=len(rr):
                raise RuntimeError(f'V5 cache length mismatch for {split_name}: {len(cached)} != {len(rr)}')
            return cached
        if not args.recompute_candidates:
            raise FileNotFoundError(f'V5 {split_name} candidate cache not found: {p}. Use same V5 CLI arguments or --recompute-candidates (expensive).')
        LOG.info('Retrieving %s candidates (%d S1)',split_name,len(rr))
        found=[]
        for start in range(0,len(rr),args.query_batch):
            found += store.retrieve(rr[start:start+args.query_batch],encoder,args.top_k)
            LOG.info('%s retrieval %d/%d',split_name,min(len(rr),start+args.query_batch),len(rr))
        if args.candidate_cache:
            joblib.dump(found,p,compress=0)
        return found
    tr_candidates=load_candidates('train',train)
    v4.candidate_recall(train,tr_candidates,truth)
    vl_candidates=load_candidates('valid',valid)
    v4.candidate_recall(valid,vl_candidates,truth)
    ceiling=v4.retrieval_oracle_f05(valid,vl_candidates,truth)
    v4.retrieval_diagnostics(valid,vl_candidates,truth,store,out/'validation_retrieval_misses_v6.tsv',300)
    LOG.info('Oracle ceiling %.5f; requested 0.987 requires oracle at least 0.987',ceiling)
    if args.retrieval_only:
        LOG.info('Retrieval-only experiment finished; classifier not trained')
        store.close()
        return
    X,y=v4.training_arrays(train,tr_candidates,truth,store,max_neg=24)
    w=positive_sample_weights(X,y,args.positive_weight)
    LOG.info('Initial training pairs: %d (positives %d, negatives %d)',len(y),int(y.sum()),int((y==0).sum()))
    device='cuda' if dev.type=='cuda' else 'cpu'
    baseline=fit_xgb(X,y,device,650,args.cpu_threads,sample_weight=w)
    # Reproduce V5's tune/report split. V6's design used the V5 audit, so V5's
    # previously reported subset is NOT an untouched holdout for V6.
    tune_refs,report_refs=train_test_split(valid,test_size=.50,random_state=SEED+1)
    tune_ids={r[0] for r in tune_refs}
    report_ids={r[0] for r in report_refs}
    def score_validation(model):
        data={}
        for st in range(0,len(valid),args.query_batch):
            data.update(v4.score_batch(valid[st:st+args.query_batch],
                     vl_candidates[st:st+args.query_batch],store,model))
            LOG.info('Validation scoring %d/%d',min(len(valid),st+args.query_batch),len(valid))
        return data
    baseline_scores=score_validation(baseline)
    tune_base={rid:p for rid,p in baseline_scores.items() if rid in tune_ids}
    bscore,bthreshold=tune(tune_base,truth)
    LOG.info('Baseline: tuning macro F0.5=%.5f threshold=%.3f',bscore,bthreshold)
    final_model=baseline
    final_scored=baseline_scores
    final_threshold=bthreshold
    chosen_tag="baseline"
    if not args.skip_mining:
        LOG.info('Mining previously unseen high-scoring negatives from ALL retrieved train candidates')
        xx,yy=mine_negatives(train,tr_candidates,truth,store,baseline,args.mining_per_entity,args.query_batch)
        LOG.info('Mined %d negative pairs',len(yy))
        if len(yy):
            X2=np.concatenate([X,xx],axis=0)
            y2=np.concatenate([y,yy],axis=0)
            del xx,yy
            gc.collect()
            # Mined pairs are deliberately given weight 1; positives receive
            # moderate additional weight, especially when one address is blank.
            w2=np.concatenate([w,np.ones(len(y2)-len(y),dtype=np.float32)])
            refined=fit_xgb(X2,y2,device,850,args.cpu_threads,sample_weight=w2)
            second = None
            if args.second_model:
                LOG.info('Training alternate-depth mined model')
                second=fit_xgb(X2,y2,device,1050,args.cpu_threads,
                               sample_weight=w2,depth=6,child_weight=1)
            del X2,y2,w2
            gc.collect()
            refined_scores=score_validation(refined)
            rscore,rthreshold=tune({rid:p for rid,p in refined_scores.items() if rid in tune_ids},truth)
            LOG.info('Mined model: tuning macro F0.5=%.5f threshold=%.3f',rscore,rthreshold)
            if rscore>bscore:
                final_model=refined
                final_scored=refined_scores
                final_threshold=rthreshold
                chosen_tag="mined"
                LOG.info('Selected mined model by TUNING subset only')
            else:
                LOG.info('Kept baseline by TUNING subset only; mining did not help here')
    if args.second_model and not args.skip_mining and 'second' in locals() and second is not None:
        alternate_scores=score_validation(second)
        a_score,a_thr=tune({rid:p for rid,p in alternate_scores.items() if rid in tune_ids},truth)
        LOG.info('Alternate-depth tuning F0.5 %.5f threshold %.3f',a_score,a_thr)
        if a_score > max(bscore,rscore):
            final_model=second;final_scored=alternate_scores;final_threshold=a_thr
            chosen_tag='alternate'
        # Blending can recover probabilities for difficult positives without
        # requiring post-hoc overrides; select solely on TUNING refs.
        blend_scored={}
        for rid,pairs in refined_scores.items():
            alt=dict(alternate_scores[rid])
            blend_scored[rid]=[(cid,(p+alt.get(cid,p))*.5) for cid,p in pairs]
        ens_score,ens_thr=tune({rid:p for rid,p in blend_scored.items() if rid in tune_ids},truth)
        LOG.info('Ensemble tuning F0.5 %.5f threshold %.3f',ens_score,ens_thr)
        if ens_score > max(bscore,rscore,a_score):
            final_scored=blend_scored;final_threshold=ens_thr;chosen_tag='ensemble'
            final_model=None # Need both models at inference time.
    rescue_cutoff=None
    if not args.no_rescue:
        hints=audit_rescue_hints(final_scored,valid,store,ceiling=max(.7,final_threshold))
        tune_data={rid:p for rid,p in final_scored.items() if rid in tune_ids}
        rescue_tune,rescue_cutoff=tuning_rescue(tune_data,truth,final_threshold,hints)
        LOG.info('Missing-address rescue: tuning %.5f cutoff=%s',rescue_tune,rescue_cutoff)
    else:
        hints=set()
    report_scores={rid:p for rid,p in final_scored.items() if rid in report_ids}
    if rescue_cutoff is None:
        report_score,report_conf=evaluate(report_scores,truth,final_threshold)
        whole_score,whole_conf=evaluate(final_scored,truth,final_threshold)
    else:
        report_score,report_conf=evaluate_with_rescue(report_scores,truth,final_threshold,hints,rescue_cutoff)
        whole_score,whole_conf=evaluate_with_rescue(final_scored,truth,final_threshold,hints,rescue_cutoff)
    LOG.info('REUSED V5 REPORT SUBSET F0.5 = %.5f at threshold %.3f; TP/FP/FN=%s',
             report_score,final_threshold,report_conf)
    LOG.info('Full 2000-entity validation F0.5 (partly tuned) = %.5f; TP/FP/FN=%s',
             whole_score,whole_conf)
    final_audit(final_scored,valid,vl_candidates,truth,store,final_threshold,
                hints,rescue_cutoff,out)
    if final_model is not None:
        final_model.save_model(str(model_dir/'v6_xgboost.json'))
    else:
        refined.save_model(str(model_dir/'v6_ensemble_refined.json'))
        second.save_model(str(model_dir/'v6_ensemble_alternate.json'))
    joblib.dump({'threshold':final_threshold,'reused_v5_report_f05':report_score,
       'full_valid_f05':whole_score,'oracle':ceiling,'chosen_model':chosen_tag,
       'rescue_cutoff':rescue_cutoff,'feature_count':X.shape[1],'settings':vars(args)},
       model_dir/'v6_metrics.joblib')
    (out/'v6_results.txt').write_text(
        f'Oracle: {ceiling:.6f}\nReused V5 report subset F0.5: {report_score:.6f}\n'
        f'Full validation F0.5 (partly tuned): {whole_score:.6f}\n'
        f'Threshold: {final_threshold:.4f}\nChosen: {chosen_tag}\n'
         f'Rescue cutoff: {rescue_cutoff}\nTP FP FN on held-out: {report_conf}\n',encoding='utf8')
    del X,y,baseline_scores,tr_candidates,vl_candidates
    gc.collect()
    store.close()
    LOG.info('V6 pipeline completed')


if __name__=='__main__':
    main()
