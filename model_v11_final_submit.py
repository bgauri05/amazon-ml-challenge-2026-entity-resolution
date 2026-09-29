#!/usr/bin/env python3
"""V11 submission finisher: V10 predictions + validated conservative SQL recovery.

Requirements: completed V10 output, V7 TRAIN / V8 TEST SQLite target DBs,
original V8 fitted classifiers. NO FAISS, encoder projection, retraining or external data.
Uses original V10 predictions as safe initial output, and improves them only when
held-out TRAIN tuning establishes a positive change. Does not claim 0.99.

Example:
 python -u model_v11_final_submit.py --root student_resource --minutes 90 --threads 8 > v11.log 2>&1
"""
from __future__ import annotations
import argparse
import collections
import csv
import json
import logging
import os
import shutil
import sqlite3
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz
import xgboost as xgb
import model_v4_hybrid as v4
import model_v6_fast as v6

LOG = logging.getLogger('v11')


def f05(pred, truth):
    if not truth:
        return float(not pred)
    return 1.25 * len(pred & truth) / (.25 * len(truth) + len(pred))


def nums(s):
    import re
    return set(re.findall(r'\d+', s))


def load_model(folder, threads):
    meta = json.loads((folder/'v8_metadata.json').read_text(encoding='utf8'))
    chosen=meta['selected']
    paths=[]
    weight=1.
    if chosen == 'v7_original': paths=[folder/'v8_selected_v7.json']
    elif chosen == 'recall': paths=[folder/'v8_recall.json']
    elif chosen == 'precision': paths=[folder/'v8_precision.json']
    elif chosen.startswith('blend_'):
        weight=int(chosen.split('_')[1])/100
        paths=[folder/'v8_precision.json',folder/'v8_recall.json']
    else: raise RuntimeError('Unknown V8 model selection '+chosen)
    models=[]
    for path in paths:
        m=xgb.XGBClassifier()
        m.load_model(str(path)); m.set_params(device='cpu',n_jobs=threads)
        models.append(m)
    return models,weight,float(meta['selection']['threshold'])


class Recover:
    def __init__(self, db_path, model_dir, threads, max_posting=18):
        if not db_path.exists():raise FileNotFoundError(str(db_path))
        self.db=sqlite3.connect(db_path.resolve().as_uri()+'?mode=ro',uri=True)
        self.db.execute('PRAGMA query_only=ON')
        self.db.execute('PRAGMA mmap_size=536870912')
        self.db.execute('PRAGMA cache_size=-32768')
        self.models,self.weight,self.threshold=load_model(model_dir,threads)
        self.cap=max_posting
        self.probes=0
        self.scored=0
        self.upgrades=0

    @lru_cache(maxsize=4096)
    def posting(self, country, field, key, target_source):
        if not key or len(key)<5:return ()
        if field not in ('name_key','addr_key'):raise ValueError(field)
        sql=('SELECT id,name,addr,country,source,name_key,addr_key FROM targets '
             f'WHERE country=? AND {field}=? AND source=? LIMIT ?')
        self.probes+=1
        rows=self.db.execute(sql,(country,key,target_source,self.cap+1)).fetchall()
        # Popular keys are highly ambiguous and cannot be treated as reliable.
        return tuple(rows) if len(rows)<=self.cap else ()

    def targets(self, ids):
        ids=list(dict.fromkeys(ids))
        out={}
        for pos in range(0,len(ids),400):
            part=ids[pos:pos+400]
            for row in self.db.execute('SELECT id,name,addr,country,source,name_key,addr_key FROM targets WHERE id IN ('+','.join('?'*len(part))+')',part):
                out[row[0]]=row
        return out

    @staticmethod
    def strong(s1, target, policy, parent=None):
        # 'exact': canonical name+address equality. Other policies still require
        # consistent street numbers and both nontrivial name/address evidence.
        sn,sa=v4.name_key(s1[1]),v4.addr_key(s1[2])
        tn,ta=target[5],target[6]
        if not sn or not tn or len(sa)<9 or len(ta)<9:return False
        a,b=nums(sa),nums(ta)
        if a and b and a.isdisjoint(b):return False
        if sn==tn and sa==ta:return True
        if policy=='exact':return False
        nr=fuzz.WRatio(sn,tn)/100
        ar=fuzz.WRatio(sa,ta)/100
        if parent is not None:
            pn,pa=parent[5],parent[6]
            # A noisy source anchor is not reliable enough to propagate from.
            if not pn or not pa or len(pa)<9 or (nums(pa) and b and nums(pa).isdisjoint(b)):
                return False
            if fuzz.WRatio(pn,tn)<93 or fuzz.WRatio(pa,ta)<88:
                return False
        if policy=='strict': return nr>=.965 and ar>=.935 and bool(a or b)
        return nr>=.93 and ar>=.91 and bool(a or b)

    def potential(self,s1,matched,known,policy):
        has2=any(x.startswith('S2-') for x in matched)
        has3=any(x.startswith('S3-') for x in matched)
        if has2 and has3:return []
        country=s1[3]
        sn=v4.name_key(s1[1]);sa=v4.addr_key(s1[2])
        missing_sources=([2,3] if not matched else ([3] if has2 else [2]))
        anchors=[]
        if matched:
            anchors=[known[mid] for mid in matched if mid in known][:2]
        proposal={}
        for dest in missing_sources:
            # S1-based indexed probes also help when the selected anchor's
            # spelling is noisy; keep both query types bounded.
            for field,key in [('addr_key',sa),('name_key',sn)]:
                if field=='addr_key' and len(key)<10:continue
                for t in self.posting(country,field,key,dest):proposal[t[0]]=(t,None)
            for anchor in anchors:
                for field,key in [('addr_key',anchor[6]),('name_key',anchor[5])]:
                    if field=='addr_key' and len(key)<10:continue
                    for t in self.posting(country,field,key,dest):proposal[t[0]]=(t,anchor)
        out=[]
        # Bound scoring even if several distinct probes return different rows.
        for cid,(target,parent) in proposal.items():
            if cid in matched:continue
            if self.strong(s1,target,policy,parent):out.append(target)
        return out[:24]

    def upgrade_batch(self, rows, predictions, policy):
        if policy=='off':return [([],[]) for _ in rows]
        wanted={mid for ms in predictions for mid in ms[:2] if mid.startswith(('S2-','S3-'))}
        known=self.targets(wanted) if wanted else {}
        pairs=[];locations=[];results=[([],[]) for _ in rows]
        for i,(s1,mids) in enumerate(zip(rows,predictions)):
            for target in self.potential(s1,mids,known,policy):
                pairs.append((s1,(target[0],target[1],target[2],target[3]),-1.))
                locations.append((i,target[0],target, s1))
        if not pairs:return results
        x=np.asarray([v6.extra_features(*p) for p in pairs],dtype=np.float32)
        scores=self.models[0].predict_proba(x)[:,1]
        if len(self.models)>1:
            scores=self.weight*scores+(1-self.weight)*self.models[1].predict_proba(x)[:,1]
        self.scored+=len(pairs)
        # Also require classifier acceptance; the fuzzy text rule cannot
        # independently force a potentially wrong match into the submission.
        # Slightly more conservative than the saved V8 threshold.
        floor=max(self.threshold,.68 if policy=='balanced' else .60)
        found=collections.defaultdict(list)
        for (i,cid,target,s1),p in zip(locations,scores):
            if p>=floor:found[i].append(cid)
        for i,mids in found.items():
            extras=list(dict.fromkeys(mids))[:5]
            results[i]=(extras,extras)
            self.upgrades+=bool(extras)
        return results

    def close(self):self.db.close()


def read_tuning(out):
    required=['fresh_validation.json','validation_scores.json','validation.json']
    if not all((out/p).exists() for p in required):
        raise FileNotFoundError('V10 validation files are required for automatic safe selection: '+', '.join(required))
    data=json.loads((out/'fresh_validation.json').read_text(encoding='utf8'))
    scores=json.loads((out/'validation_scores.json').read_text(encoding='utf8'))
    report=json.loads((out/'validation.json').read_text(encoding='utf8'))
    rows=data['rows'];truth={k:set(v) for k,v in data['truth'].items()}
    threshold=float(report['threshold'])
    ids=[r[0] for r in rows]
    missing=set(ids)-set(scores)
    if missing:raise RuntimeError('V10 validation_scores missing '+str(len(missing))+' entities')
    baseline={k:list(dict.fromkeys(cid for cid,p in zip(*scores[k]) if p>=threshold)) for k in ids}
    permutation=np.random.default_rng(20260928).permutation(len(rows))
    tune=[rows[int(i)][0] for i in permutation[:len(rows)//2]]
    report_ids=[rows[int(i)][0] for i in permutation[len(rows)//2:]]
    return rows,truth,baseline,tune,report_ids


def validate_policy(a):
    rows,truth,baseline,tune,hold=read_tuning(a.baseline)
    rec=Recover(a.root/'artifacts_v7_fresh/train_targets.sqlite',a.root/'models_v8',a.threads,a.max_posting)
    byid={r[0]:r for r in rows};variants={'off':baseline}
    try:
        for policy in ('exact','strict','balanced'):
            result={k:list(v) for k,v in baseline.items()}
            start=time.monotonic()
            for j in range(0,len(rows),a.batch):
                batch=rows[j:j+a.batch]
                extra=rec.upgrade_batch(batch,[baseline[r[0]] for r in batch],policy)
                for r,(new,_) in zip(batch,extra):result[r[0]].extend(c for c in new if c not in result[r[0]])
            variants[policy]=result
            LOG.info('Validation policy %s finished in %.1fs',policy,time.monotonic()-start)
    finally:rec.close()
    def val(d,subset):return float(np.mean([f05(set(d[k]),truth[k]) for k in subset]))
    evaluated={policy:{'tune':round(val(data,tune),6),'heldout':round(val(data,hold),6)} for policy,data in variants.items()}
    base=evaluated['off']['tune']
    # Require >= two extra perfect S1 results per 1000 entities to pay for a
    # change. Never select on heldout; report it as diagnostic only.
    admissible=[p for p in ('exact','strict','balanced') if evaluated[p]['tune']>=base+.002]
    selected=max(admissible,key=lambda p:evaluated[p]['tune']) if admissible else 'off'
    report={'selected_policy':selected,'policies':evaluated,
            'original_v10_threshold':json.loads((a.baseline/'validation.json').read_text())['threshold'],
            'note':'Selection only on fixed 50% tuning split. Holdout has been inspected in earlier experiments. Test France has no labels; no score promise.'}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'v11_validation.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    LOG.info('V11 VALIDATION %s',json.dumps(report))
    return selected


def rows_from_partition(root, country):
    path=root/'artifacts_v8_test/s1_by_country'/f'{country}.tsv'
    if not path.exists():raise FileNotFoundError(f'V8 S1 country partition not found: {path}')
    with path.open(encoding='utf8',newline='') as f:
        yield from (tuple(r) for r in csv.reader(f,delimiter='\t'))


def finish(a,policy):
    source_match=a.baseline/'matching_results.tsv'
    source_cand=a.baseline/'candidate_pairs.tsv'
    if not source_match.exists() or not source_cand.exists():
        raise FileNotFoundError('Need both completed V10 submission files in '+str(a.baseline))
    a.output.mkdir(parents=True,exist_ok=True)
    final_m=a.output/'matching_results.tsv';final_c=a.output/'candidate_pairs.tsv'
    # Create complete and immediately usable baseline FIRST. Working results
    # replace them only after successful end-to-end processing.
    # Always reset both final outputs from the same known-good V10 snapshot.
    # The separate *.working files preserve this fallback during processing.
    LOG.info('Securing full V10 baseline in V11 directory')
    for src,dst in ((source_match,final_m),(source_cand,final_c)):
        temp=dst.with_name(dst.name+'.baseline_tmp')
        shutil.copyfile(src,temp)
        os.replace(temp,dst)
    if policy=='off':
        LOG.info('No tested recovery improvement; V11 retains validated V10 baseline.')
        LOG.info('SUBMISSION READY (unchanged V10 baseline): %s',a.output)
        return
    work=a.root/'artifacts_v8_test'
    counts=json.loads((work/'s1_by_country/complete.json').read_text(encoding='utf8'))
    expected=sum(counts.values())
    m_tmp=a.output/'matching_results.tsv.working'
    c_tmp=a.output/'candidate_pairs.tsv.working'
    rec=Recover(work/'train_targets.sqlite',a.root/'models_v8',a.threads,a.max_posting)
    budget=time.monotonic()+60*a.minutes
    active=True
    num=0
    recovered=0
    countries=sorted(counts)
    try:
        with source_match.open(encoding='utf8',newline='') as sm, source_cand.open(encoding='utf8',newline='') as sc, m_tmp.open('w',encoding='utf8',newline='') as om, c_tmp.open('w',encoding='utf8',newline='') as oc:
            mread=csv.reader(sm,delimiter='\t');cread=csv.reader(sc,delimiter='\t')
            mw=csv.writer(om,delimiter='\t',lineterminator='\n');cw=csv.writer(oc,delimiter='\t',lineterminator='\n')
            mh=next(mread);ch=next(cread)
            if mh!=['source1_entity_id','matched_entity_ids'] or ch!=['source1_entity_id','candidate_entity_ids']:
                raise RuntimeError('V10 file headers do not match required format')
            mw.writerow(mh);cw.writerow(ch)
            for country in countries:
                input_s1=rows_from_partition(a.root,country)
                processed=0
                while processed<counts[country]:
                    s1=[];mb=[];cb=[]
                    for _ in range(min(a.batch,counts[country]-processed)):
                        r=next(input_s1)
                        m=next(mread);c=next(cread)
                        if r[0]!=m[0] or r[0]!=c[0]:
                            raise RuntimeError(f'V10 output not aligned to V8 country S1 partitions: country={country} s1={r[0]} match={m[0]} candidates={c[0]}. Your original V10 output was NOT modified.')
                        s1.append(r);mb.append(list(dict.fromkeys(filter(None,m[1].split(',')))));cb.append(list(dict.fromkeys(filter(None,c[1].split(',')))))
                    if active and time.monotonic() < budget:
                        gained=rec.upgrade_batch(s1,mb,policy)
                    else:
                        if active:LOG.warning('Budget reached; copying remaining V10 predictions unchanged.')
                        active=False;gained=[([],[]) for _ in s1]
                    for r,mids,cids,(extra,scored) in zip(s1,mb,cb,gained):
                        recovered+=sum(cid not in mids for cid in extra)
                        mids=list(dict.fromkeys(mids+extra))
                        # Every added prediction was actually scored by the
                        # classifier and must appear in candidate_pairs.tsv.
                        cids=list(dict.fromkeys(cids+scored+mids))
                        mw.writerow((r[0],','.join(mids)))
                        cw.writerow((r[0],','.join(cids)))
                    processed+=len(s1);num+=len(s1)
                    if num%10000<a.batch:
                        LOG.info('V11 %d/%d rows; %d extra predictions; SQL probes=%d; ML pairs=%d',num,expected,recovered,rec.probes,rec.scored)
            if next(mread,None) is not None or next(cread,None) is not None:
                raise RuntimeError('Unexpected extra rows in V10 outputs')
        if num!=expected:raise RuntimeError(f'Incomplete: {num}/{expected}')
        # Replacing outputs one after another is fine only after BOTH working
        # outputs are complete. On interruption original V10 copies survive.
        os.replace(m_tmp,final_m);os.replace(c_tmp,final_c)
        (a.output/'v11_finish.json').write_text(json.dumps({'policy':policy,'total_rows':num,'extra_predictions':recovered,'probes':rec.probes,'scored_pairs':rec.scored,'budget_minutes':a.minutes},indent=2))
        LOG.info('SUBMISSION READY: %s (new predicted IDs=%d)',a.output,recovered)
    finally:rec.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('student_resource'))
    p.add_argument('--baseline',type=Path,default=Path('student_resource/output_v10'))
    p.add_argument('--output',type=Path,default=Path('student_resource/output_v11'))
    p.add_argument('--batch',type=int,default=600)
    p.add_argument('--minutes',type=float,default=75.)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--max-posting',type=int,default=18)
    p.add_argument('--policy',choices=['auto','off','exact','strict','balanced'],default='auto',help='Default auto checks tuning subset before using recovery. Do not force other modes without evidence.')
    a=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',stream=sys.stdout)
    if min(a.batch,a.minutes,a.threads,a.max_posting)<=0:p.error('Positive batch/minutes/threads/max-posting required')
    if a.output.resolve()==a.baseline.resolve():
        p.error('--output must differ from --baseline')
    a.output.mkdir(parents=True,exist_ok=True)
    if not (a.baseline/'matching_results.tsv').exists() or not (a.baseline/'candidate_pairs.tsv').exists():
        p.error('Completed V10 output files required in '+str(a.baseline))
    # Preserve original complete V10 output BEFORE any optional validation.
    for filename in ('matching_results.tsv','candidate_pairs.tsv'):
        dst=a.output/filename
        tmp=dst.with_name(dst.name+'.baseline_tmp')
        shutil.copyfile(a.baseline/filename,tmp)
        os.replace(tmp,dst)
    LOG.info('SECURE BASELINE: %s contains both full V10 outputs',a.output)
    if a.policy=='auto':
        try: policy=validate_policy(a)
        except (FileNotFoundError,MemoryError) as exc:
            LOG.warning('Validation unavailable: %s. Retaining V10 baseline; no untested modifications.',exc)
            policy='off'
    else:policy=a.policy
    finish(a,policy)
    LOG.info('Run competition validator on %s; do not infer score from format PASS.',a.output)


if __name__=='__main__':main()
