#!/usr/bin/env python3
"""V10: bounded exact + three-channel ANN, identical validation/test decisions.
Reuses V7 encoders/indexes and V8 classifiers. No external data or score promise.
"""
import argparse, csv, gc, hashlib, json, logging, os, re, sqlite3, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor
from rapidfuzz import fuzz
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import faiss, joblib, numpy as np, torch
import model_v4_hybrid as v4
import model_v6_fast as v6
import model_v7_fresh as v7
import model_v8_final as v8
import model_v9_deadline_safe as v9
LOG=logging.getLogger('v10')

def atomic_json(path, obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2),encoding='utf8')
    os.replace(tmp,path)

def metric(pred, actual):
    if not actual: return float(not pred)
    return 1.25*len(pred & actual)/(.25*len(actual)+len(pred))


def worker_init(dbpath):
    global WORKER_DB
    WORKER_DB=sqlite3.connect(Path(dbpath).resolve().as_uri()+'?mode=ro',uri=True)
    WORKER_DB.execute('PRAGMA mmap_size=2147418112')
    WORKER_DB.execute('PRAGMA cache_size=-32768')
    WORKER_DB.execute('PRAGMA query_only=ON')
    torch.set_num_threads(1)
    # Several of the 82 features reuse exactly the same fuzzy comparisons.
    for name in ('ratio','WRatio','partial_ratio','token_set_ratio','token_sort_ratio'):
        setattr(fuzz,name,lru_cache(maxsize=8192)(getattr(fuzz,name)))
    v4.name_key=lru_cache(maxsize=30000)(v4.name_key)
    v4.addr_key=lru_cache(maxsize=30000)(v4.addr_key)
    for name in ('clean_alnum','initials','digit_tokens','char_ngrams','ocr_fold'):
        setattr(v6,name,lru_cache(maxsize=30000)(getattr(v6,name)))

def lexical_job(payload):
    from types import SimpleNamespace
    rows,df,cap,probe_cap=payload
    e=Engine.__new__(Engine);e.db=WORKER_DB;e.df=df;e.a=SimpleNamespace(lexical=cap,probe_cap=probe_cap);e.pool=None
    return e.lexical(rows)

def feature_job(pairs):
    return np.asarray([v6.extra_features(a,b,sim) for a,b,sim in pairs],dtype=np.float32)

class Engine:
    def __init__(self,a,work):
        self.a=a; self.work=work; self.country=None; self.indexes=[]
        self.db=sqlite3.connect((work/'train_targets.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
        self.db.execute("CREATE VIRTUAL TABLE temp.vocab USING fts5vocab(main, targets_fts, 'row')")
        self.db.execute('PRAGMA mmap_size=2147418112')
        self.db.execute('PRAGMA cache_size=-131072')
        self.db.execute('PRAGMA query_only=ON')
        freq=a.output/('frequencies_'+work.name+'.joblib')
        if freq.exists():self.df=joblib.load(freq)
        else:
            LOG.info('Caching text-index token frequencies once: %s',work)
            self.df=dict(self.db.execute('SELECT term,doc FROM vocab'))
            joblib.dump(self.df,freq,compress=0)
        LOG.info('Loaded %d token frequencies',len(self.df))
        self.pool=ProcessPoolExecutor(max_workers=a.workers,initializer=worker_init,initargs=(str(work/'train_targets.sqlite'),)) if a.workers>1 else None
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enc=v8.load_v7_encoders(a.root/'artifacts_v7_fresh',a,device)
        self.models,self.weight,self.threshold=v9.model_bundle(a.root/'models_v8',a.threads)
    def activate(self,country):
        if self.country==country:return
        self.indexes.clear();gc.collect()
        for source in (2,3):
            tag=f'{source}_{country}'
            field_ids=None
            for field in ('combined','name','addr'):
                folder=self.work/('combined_indexes' if field=='combined' else 'field_indexes')
                path=folder/(f'{tag}.faiss' if field=='combined' else f'{field}_{tag}.faiss')
                ix=faiss.read_index(str(path));ix.hnsw.efSearch=self.a.ef
                if field=='combined': ids=joblib.load(folder/f'{tag}.ids.joblib')
                else:
                    if field_ids is None:field_ids=joblib.load(folder/f'{tag}.ids.joblib')
                    ids=field_ids
                if ix.ntotal!=len(ids):raise RuntimeError(f'Index/ID mismatch: {path}')
                self.indexes.append((field,ix,ids))
        self.country=country;LOG.info('Loaded six indexes for %s',country)
    def lexical(self,rows):
        if not self.a.lexical:return [{} for _ in rows]
        if self.pool:
            jobs=[]
            for start in range(0,len(rows),50):
                part=rows[start:start+50]
                tokens={t for r in part for text in (r[1],r[2]) for t in re.findall(r'[a-z0-9]+',text) if len(t)>=3}
                jobs.append((part,{t:self.df.get(t,0) for t in tokens},self.a.lexical,self.a.probe_cap))
            return [hits for group in self.pool.map(lexical_job,jobs) for hits in group]
        tokenized=[];needed=set()
        for r in rows:
            fields=[]
            for text in (r[1],r[2]):
                tokens=set(re.findall(r'[a-z0-9]+',text))
                tokens={t for t in tokens if len(t)>=3}
                needed.update(tokens);fields.append(tokens)
            tokenized.append(fields)
        missing=list(needed-self.df.keys())
        for start in range(0,len(missing),400):
            part=missing[start:start+400]
            found=dict(self.db.execute('SELECT term,doc FROM vocab WHERE term IN ('+','.join('?'*len(part))+')',part))
            self.df.update({t:found.get(t,0) for t in part})
        out=[]
        sql=('SELECT t.id,t.name,t.addr,t.country FROM targets_fts '
             'CROSS JOIN targets t ON t.rowid=targets_fts.rowid '
             'WHERE targets_fts MATCH ? AND t.country=? LIMIT '+str(self.a.probe_cap))
        for r,fields in zip(rows,tokenized):
            found={}
            for field,tokens in zip(('name','addr'),fields):
                ranked=sorted((t for t in tokens if self.df[t]>0),key=lambda t:(self.df[t],t))[:4]
                probes=[]
                for token in ranked[:2]:
                    if self.df[token]<=120:probes.append([token])
                if len(ranked)>=2:probes.append(ranked[:2])
                if len(ranked)>=3:probes.extend([[ranked[0],ranked[2]],[ranked[1],ranked[2]]])
                complete_singles={ts[0] for ts in probes if len(ts)==1 and self.df[ts[0]]<=self.a.probe_cap}
                for terms in probes:
                    # A complete singleton posting list already contains these intersections.
                    if len(terms)>1 and complete_singles.intersection(terms):continue
                    query=' AND '.join(field+':"'+t+'"' for t in terms)
                    for target in self.db.execute(sql,(query,r[3])):found[target[0]]=target
            # Bounded final lexical shortlist; only this shortlist reaches the ML model.
            def rank(b):
                n=fuzz.WRatio(r[1],b[1])/100.;ad=fuzz.WRatio(r[2],b[2])/100. if r[2] and b[2] else 0.
                return max(n,ad)+.35*min(n,ad)
            out.append({b[0]:b for b in sorted(found.values(),key=rank,reverse=True)[:self.a.lexical]})
        # Frequencies are immutable for this target corpus.
        return out
    def score(self,rows):
        self.activate(rows[0][3])
        if any(r[3]!=self.country for r in rows):raise ValueError('Mixed country batch')
        began=time.monotonic()
        exact=v9.exact_batch(self.db,rows,self.a.exact_cap)
        pairs=[{cid:-1. for cid in x} for x in exact]
        targets={cid:(cid,n,ad,rows[i][3]) for i,x in enumerate(exact) for cid,(n,ad) in x.items()}
        lex=self.lexical(rows)
        for i,hits in enumerate(lex):
            for cid,b in hits.items():
                pairs[i].setdefault(cid,-1.);targets[cid]=b
        t1=time.monotonic()
        for field in ('combined','name','addr'):
            vectors=np.ascontiguousarray(self.enc[field].transform(rows,batch_size=self.a.batch),dtype=np.float32)
            for f,ix,ids in self.indexes:
                if f!=field:continue
                sims,labs=ix.search(vectors,min(self.a.k,ix.ntotal))
                for i,(ss,ll) in enumerate(zip(sims,labs)):
                    for sim,label in zip(ss,ll):
                        if label<0:continue
                        cid=ids[int(label)]
                        if field=='combined':pairs[i][cid]=float(sim)
                        else:pairs[i].setdefault(cid,-1.)
        t2=time.monotonic()
        missing=list({cid for p in pairs for cid in p if cid not in targets})
        for start in range(0,len(missing),500):
            ids=missing[start:start+500]
            for r in self.db.execute('SELECT id,name,addr,country FROM targets WHERE id IN ('+','.join('?'*len(ids))+')',ids):targets[r[0]]=r
        feats=[];flat=[];spans=[]
        for r,p in zip(rows,pairs):
            start=len(flat)
            for cid,sim in p.items():
                if cid not in targets:raise RuntimeError(f'Missing metadata: {cid}')
                feats.append((r,targets[cid],sim));flat.append(cid)
            spans.append((start,len(flat)))
        if feats:
            if self.pool:
                x=np.concatenate(list(self.pool.map(feature_job,[feats[i:i+400] for i in range(0,len(feats),400)])))
            else:x=feature_job(feats)
            scores=self.models[0].predict_proba(x)[:,1]
            if len(self.models)>1:scores=self.weight*scores+(1-self.weight)*self.models[1].predict_proba(x)[:,1]
        else:scores=np.empty(0)
        elapsed=time.monotonic()-began
        LOG.info('BATCH %s n=%d pairs=%d exact=%.2f ANN=%.2f score=%.2f rate=%.1f/s',self.country,len(rows),len(flat),t1-began,t2-t1,time.monotonic()-t2,len(rows)/max(elapsed,.001))
        return [(flat[lo:hi],scores[lo:hi].tolist()) for lo,hi in spans],elapsed

def settings(a):return {'k':a.k,'ef':a.ef,'exact_cap':a.exact_cap,'lexical':a.lexical,'probe_cap':a.probe_cap,'schema':3}

def validation(a):
    cache=a.output/'fresh_validation.json'
    if cache.exists():data=json.loads(cache.read_text(encoding='utf8'))
    else:
        LOG.info('Sampling fresh validation; excluding ALL original 10000 sampled entities')
        old=v4.sample_records(a.root/'dataset/train/train_source1.tsv',10000,seed=2026)
        excluded={r[0] for r in old}
        fresh=v4.sample_records(a.root/'dataset/train/train_source1.tsv',a.valid_rows+100,seed=20260927)
        rows=[r for r in fresh if r[0] not in excluded][:a.valid_rows]
        if len(rows)!=a.valid_rows:raise RuntimeError('Insufficient fresh validation rows')
        truth=v7.selected_truth(a.root/'dataset/train/train_ground_truth.tsv',{r[0] for r in rows})
        data={'rows':rows,'truth':{k:sorted(v) for k,v in truth.items()}}
        atomic_json(cache,data)
    rows=data['rows'];truth={k:set(v) for k,v in data['truth'].items()}
    # Split fixed before looking at labels or model outputs.
    order=np.random.default_rng(20260928).permutation(len(rows))
    tuning={rows[i][0] for i in order[:len(rows)//2]}
    engine=Engine(a,a.root/'artifacts_v7_fresh');scored={};timings=[]
    for country in sorted({r[3] for r in rows}):
        group=[r for r in rows if r[3]==country]
        for start in range(0,len(group),a.batch):
            batch=group[start:start+a.batch];result,elapsed=engine.score(batch);timings.append(elapsed)
            scored.update({r[0]:s for r,s in zip(batch,result)})
    def evaluate(ids,t):
        vals=[];tp=fp=fn=0
        for rid in ids:
            cs,ps=scored[rid];pred={c for c,p in zip(cs,ps) if p>=t};gt=truth[rid]
            vals.append(metric(pred,gt));tp+=len(pred&gt);fp+=len(pred-gt);fn+=len(gt-pred)
        return {'macro_f05':float(np.mean(vals)),'tp':tp,'fp':fp,'fn':fn,'n':len(vals)}
    grid=np.arange(.3,.981,.01)
    threshold=max(grid,key=lambda t:evaluate(tuning,float(t))['macro_f05'])
    report=set(scored)-tuning
    ceilings=[metric(set(scored[r][0])&truth[r],truth[r]) for r in report]
    retrieved=sum(len(set(scored[r][0])&truth[r]) for r in report)
    result={'settings':settings(a),'threshold':float(threshold),'tuning':evaluate(tuning,threshold),
            'heldout':evaluate(report,threshold),'heldout_oracle':float(np.mean(ceilings)),
            'heldout_pair_recall':retrieved/max(1,sum(len(truth[r]) for r in report)),
            'by_country':{c:evaluate([r[0] for r in rows if r[3]==c and r[0] in report],threshold) for c in sorted({r[3] for r in rows})},
            'notes':'S1 entities excluded from original 10000 classifier sample; encoders used unlabeled corpus. Reporting subset has now been inspected across retrieval experiments, so is not pristine. No France labels available.'}
    atomic_json(a.output/'validation.json',result)
    atomic_json(a.output/'validation_scores.json',scored)
    LOG.info('VALIDATION %s',json.dumps(result))

def inference(a):
    report=json.loads((a.output/'validation.json').read_text())
    if report['settings']!=settings(a):raise RuntimeError('Settings differ from validated configuration')
    threshold=report['threshold'];work=a.root/'artifacts_v8_test'
    counts=json.loads((work/'s1_by_country/complete.json').read_text())
    # Verify supplied source file sizes against existing test index manifest.
    manifest=json.loads((work/'test_index_settings.json').read_text())
    for i,(_,size,_) in enumerate(manifest['sources'],1):
        if (a.root/f'dataset/test/test_source{i}.tsv').stat().st_size!=size:raise RuntimeError('Test source size changed')
    engine=Engine(a,work);rates={};progress=a.output/'parts';progress.mkdir(exist_ok=True)
    signature={**settings(a),'threshold':threshold}
    marker=progress/'settings.json'
    if a.mode=='test':
        if marker.exists() and json.loads(marker.read_text())!=signature and (list(progress.glob('*.checkpoint.json')) or list(progress.glob('*.done.json'))):
            raise RuntimeError('Output settings changed; choose a new output directory')
        atomic_json(marker,signature)
    for country in sorted(counts,key=lambda c:-counts[c]):
        done=progress/f'{country}.done.json';ck=progress/f'{country}.checkpoint.json'
        mp=progress/f'{country}.matching.tsv';cp=progress/f'{country}.candidates.tsv'
        if a.mode=='test' and done.exists():continue
        checkpoint=json.loads(ck.read_text()) if ck.exists() and a.mode=='test' else {'rows':0,'m':0,'c':0}
        completed=checkpoint['rows'];measured=0;seconds=0
        if a.mode=='test':
            mf=mp.open('r+b' if mp.exists() else 'w+b');cf=cp.open('r+b' if cp.exists() else 'w+b')
            mf.truncate(checkpoint['m']);cf.truncate(checkpoint['c']);mf.seek(0,2);cf.seek(0,2)
        try:
            with (work/'s1_by_country'/f'{country}.tsv').open(encoding='utf8',newline='') as source:
                reader=csv.reader(source,delimiter='\t')
                for _ in range(completed):next(reader)
                while True:
                    rows=[]
                    limit=min(a.batch,a.benchmark_rows-measured) if a.mode=='benchmark' else a.batch
                    if limit<=0:break
                    for _ in range(limit):
                        try:rows.append(tuple(next(reader)))
                        except StopIteration:break
                    if not rows:break
                    scored,elapsed=engine.score(rows);measured+=len(rows);seconds+=elapsed
                    if a.mode=='test':
                        for r,(ids,probs) in zip(rows,scored):
                            selected=[c for c,p in zip(ids,probs) if p>=threshold]
                            mf.write((r[0]+'\t'+','.join(selected)+'\n').encode('utf8'))
                            cf.write((r[0]+'\t'+','.join(ids)+'\n').encode('utf8'))
                        mf.flush();cf.flush();os.fsync(mf.fileno());os.fsync(cf.fileno())
                        completed+=len(rows)
                        atomic_json(ck,{'rows':completed,'m':mf.tell(),'c':cf.tell()})
                        if completed%10000<a.batch:LOG.info('PROGRESS %s %d/%d',country,completed,counts[country])
        finally:
            if a.mode=='test':mf.close();cf.close()
        rates[country]=measured/max(seconds,.001)
        if a.mode=='test':
            if completed!=counts[country]:raise RuntimeError('Incomplete country')
            atomic_json(done,{'rows':completed})
    if a.mode=='benchmark':
        estimate=sum(counts[c]/rates[c] for c in counts)
        result={'rates':rates,'estimated_active_hours':estimate/3600,'note':'Small benchmark excludes index loading, final writing and validation; allow margin.'}
        atomic_json(a.output/'benchmark.json',result);LOG.info('BENCHMARK %s',json.dumps(result));return
    for name,suffix,header in [('matching_results.tsv','matching','matched_entity_ids'),('candidate_pairs.tsv','candidates','candidate_entity_ids')]:
        import shutil
        temp=a.output/(name+'.tmp')
        with temp.open('wb') as dst:
            dst.write(('source1_entity_id\t'+header+'\n').encode())
            for country in sorted(counts):
                with (progress/f'{country}.{suffix}.tsv').open('rb') as src:shutil.copyfileobj(src,dst)
        os.replace(temp,a.output/name)
    LOG.info('SUBMISSION READY %s',a.output)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['validate','benchmark','test'],required=True)
    p.add_argument('--root',type=Path,default=Path('student_resource'))
    p.add_argument('--output',type=Path,default=Path('student_resource/output_v10'))
    p.add_argument('--probe-cap',type=int,default=40);p.add_argument('--workers',type=int,default=8);p.add_argument('--lexical',type=int,default=16)
    p.add_argument('--k',type=int,default=4);p.add_argument('--ef',type=int,default=48)
    p.add_argument('--exact-cap',type=int,default=10);p.add_argument('--batch',type=int,default=400)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--valid-rows',type=int,default=2000)
    p.add_argument('--benchmark-rows',type=int,default=2000)
    a=p.parse_args();a.dim=64;a.svd_samples=60000
    warnings.filterwarnings('ignore',message='Sparse invariant checks are implicitly disabled.*')
    if min(a.k,a.ef,a.exact_cap,a.batch,a.threads,a.valid_rows,a.benchmark_rows)<=0:p.error('Counts must be positive')
    a.output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',stream=sys.stdout)
    faiss.omp_set_num_threads(a.threads);torch.set_num_threads(a.threads)
    v4.name_key=lru_cache(maxsize=100000)(v4.name_key);v4.addr_key=lru_cache(maxsize=100000)(v4.addr_key)
    # Cache pure per-string transformations reused across candidate pairs.
    for name in ('clean_alnum','initials','digit_tokens','char_ngrams','ocr_fold'):
        setattr(v6,name,lru_cache(maxsize=30000)(getattr(v6,name)))
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    if a.mode=='validate':validation(a)
    else:inference(a)
if __name__=='__main__':main()
