#!/usr/bin/env python3
"""V13: FULL supervised business-pair LightGBM + hybrid retrieval + test submission.

Runs on the user's local/AWS dataset. Never uses test labels or outside records.
A fresh hash-held-out part is used for threshold selection; --refit-all (default)
then fits final model on ALL labeled S1 records, using ALL positive links and
within-batch/SQL hard negatives. New retrieval supplements existing V10 blocking.
No leaderboard score is promised; this file has not run on the user's full data.
"""
from __future__ import annotations
import argparse, csv, gc, hashlib, json, logging, os, random, re, shutil, sqlite3, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
from rapidfuzz import fuzz
try:
    from unidecode import unidecode
except ImportError as exc:
    raise SystemExit('Missing Unidecode. Run: python -m pip install Unidecode') from exc

LOG = logging.getLogger('v13')
FIELDS = ['entity_id','business_name','business_address','country']
STOPS = set('inc llc ltd limited private pvt company corporation corp incorporated and the com www co llp plc'.split())
ADDR_ALIASES = {'street':'st','st':'st','road':'rd','rd':'rd','avenue':'ave','ave':'ave',
                'boulevard':'blvd','drive':'dr','apartment':'apt','suite':'ste',
                'lane':'ln','highway':'hwy','nagar':'ngr','saint':'st',
                'building':'bldg','floor':'fl','marg':'rd','rue':'rue'}
INDEX_STOPS=set('inc incorporated llc l ltd limited pvt private corp corporation co company llp plc sa sarl srl com www the'.split())
STOP_TOKENS = STOPS | set('business service services enterprise enterprises building floor near city state india france united states street road avenue company'.split())
FCOUNT = 27

@lru_cache(maxsize=120000)
def norm(raw):
    s = unidecode(str(raw or '').replace('&',' and ')).lower()
    return ' '.join(re.findall(r'[a-z0-9]+',s))

@lru_cache(maxsize=120000)
def nkey(s):
    return ' '.join(w for w in norm(s).split() if w not in STOPS)

@lru_cache(maxsize=120000)
def indexed_nkey(s):
    # Original V4 stored name_key with this specific legal-word list.
    return ' '.join(w for w in norm(s).split() if w not in INDEX_STOPS)

@lru_cache(maxsize=120000)
def akey(s):
    return ' '.join(ADDR_ALIASES.get(w,w) for w in norm(s).split())

@lru_cache(maxsize=120000)
def numbers(s):
    return frozenset(x.lstrip('0') or '0' for x in re.findall(r'\d+',s))

@lru_cache(maxsize=120000)
def tokens(s):
    return frozenset(s.split())

def source(row):
    return (row[0], norm(row[1]), norm(row[2]), norm(row[3]))

def stream_source(path):
    with Path(path).open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f,delimiter='\t')
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f'Unexpected source columns in {path}: {reader.fieldnames}')
        for r in reader:
            yield source(tuple(r[k] for k in FIELDS))

def chunks(iterable,n):
    block=[]
    for x in iterable:
        block.append(x)
        if len(block)>=n:
            yield block;block=[]
    if block:yield block

def dbopen(p,readonly=False):
    if not Path(p).exists() and readonly:raise FileNotFoundError(p)
    db=sqlite3.connect(Path(p).resolve().as_uri()+'?mode=ro' if readonly else str(p),uri=readonly)
    db.execute('PRAGMA cache_size=-65536')
    db.execute('PRAGMA mmap_size=536870912')
    if readonly:db.execute('PRAGMA query_only=ON')
    else:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous=NORMAL')
    return db

def atomic(path,obj):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2),encoding='utf8');os.replace(tmp,path)

def get_targets(db,ids):
    result={}
    for part in chunks(dict.fromkeys(x for x in ids if x),450):
        query='SELECT id,name,addr,country FROM targets WHERE id IN ('+','.join('?'*len(part))+')'
        result.update({x[0]:tuple(x) for x in db.execute(query,part)})
    return result

def feat(a,b):
    """27 cheap, country-agnostic signals; feature schema identical train/test."""
    an=nkey(a[1]);bn=nkey(b[1]);aa=akey(a[2]);ba=akey(b[2])
    xn=tokens(an);yn=tokens(bn);xa=tokens(aa);ya=tokens(ba)
    na=numbers(aa);nb=numbers(ba)
    nr=fuzz.ratio(an,bn)/100 if an and bn else 0.
    ar=fuzz.ratio(aa,ba)/100 if aa and ba else 0.
    ns=fuzz.token_set_ratio(an,bn)/100 if an and bn else 0.
    ads=fuzz.token_set_ratio(aa,ba)/100 if aa and ba else 0.
    nc=an.replace(' ','');bc=bn.replace(' ','')
    def jac(x,y):return len(x&y)/len(x|y) if x and y else 0.
    x=[nr,ar,ns,ads,
       fuzz.WRatio(an,bn)/100 if an and bn else 0.,
       fuzz.WRatio(aa,ba)/100 if aa and ba else 0.,
       fuzz.ratio(nc,bc)/100 if nc and bc else 0.,
       fuzz.partial_ratio(an,bn)/100 if an and bn else 0.,
       jac(xn,yn),jac(xa,ya),
       float(bool(an) and an==bn),float(bool(aa) and aa==ba),
       float(bool(nc) and nc==bc),
       float(bool(na and nb) and na==nb),
       float(bool(na and nb) and not na.isdisjoint(nb)),
       float(bool(na and nb) and na.isdisjoint(nb)),
       float(not b[2]),
       float(bool(an and bn) and (an in bn or bn in an)),
       float(bool(aa and ba) and (aa in ba or ba in aa)),
       min(abs(len(an)-len(bn))/max(1,max(len(an),len(bn))),1.),
       min(abs(len(aa)-len(ba))/max(1,max(len(aa),len(ba))),1.),
       float(bool(xn and yn) and next(iter(an.split()),'')==next(iter(bn.split()),'')),
       float(bool(xn and yn) and an.split()[-1]==bn.split()[-1]),
       float(bool(a[1]) and a[1]==b[1]),
       float(bool(a[2]) and a[2]==b[2]),
       float(b[0].startswith('S2-')),
       float(bool(aa) and bool(ba))]
    assert len(x)==FCOUNT
    return x

def feature_block(pairs):
    return np.asarray([feat(a,b) for a,b in pairs],dtype=np.float32).reshape((-1,FCOUNT))

class FeatureComputer:
    def __init__(self,workers):
        self.pool=ProcessPoolExecutor(max_workers=workers) if workers>1 else None
    def compute(self,pairs):
        if not pairs:return np.empty((0,FCOUNT),np.float32)
        if not self.pool or len(pairs)<400:return feature_block(pairs)
        blocks=[pairs[i:i+450] for i in range(0,len(pairs),450)]
        return np.concatenate(list(self.pool.map(feature_block,blocks, chunksize=2)))
    def close(self):
        if self.pool:self.pool.shutdown()

def ensure_targets(root,work,split):
    previous=(root/('artifacts_v7_fresh' if split=='train' else 'artifacts_v8_test')/'train_targets.sqlite')
    if previous.exists():
        with dbopen(previous,True) as db:
            n=db.execute('SELECT count(*) FROM targets').fetchone()[0]
            if not n:raise RuntimeError(f'Empty target database: {previous}')
            idx={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='targets'")}
            if not {'ix_target_name','ix_target_addr'}.issubset(idx):
                LOG.warning('Existing DB missing exact index names; exact retrieval may be slower')
            LOG.info('Reusing %s target DB: %d rows',split,n)
        return previous
    dest=work/f'{split}_targets.sqlite';marker=work/f'{split}_targets.complete'
    if marker.exists():return dest
    LOG.info('Building independent %s target SQLite (first run only)',split)
    if dest.exists():dest.unlink()
    db=dbopen(dest)
    db.execute('CREATE TABLE targets(id TEXT PRIMARY KEY,name TEXT,addr TEXT,country TEXT,source INT,name_key TEXT,addr_key TEXT)')
    count=0
    for s in (2,3):
        path=root/'dataset'/split/f'{split}_source{s}.tsv'
        for rows in chunks(stream_source(path),20000):
            db.executemany('INSERT INTO targets VALUES(?,?,?,?,?,?,?)',
                           [(r[0],r[1],r[2],r[3],s,nkey(r[1]),akey(r[2])) for r in rows])
            count+=len(rows)
            if count%1000000<20000:LOG.info('Loaded %s targets: %d',split,count)
            db.commit()
    db.execute('CREATE INDEX ix_target_name ON targets(country,name_key)')
    db.execute('CREATE INDEX ix_target_addr ON targets(country,addr_key)')
    db.commit()
    LOG.info('Building FTS5 retrieval index for independent %s DB',split)
    db.execute("CREATE VIRTUAL TABLE targets_fts USING fts5(name,addr,tokenize='unicode61 remove_diacritics 2')")
    # Run within SQLite; no massive Python-side object list.
    db.execute('INSERT INTO targets_fts(rowid,name,addr) SELECT rowid,name,addr FROM targets')
    db.commit();db.close();marker.write_text(str(count));return dest

def ensure_train_sources(root,work):
    target=work/'train_s1_truth.sqlite';marker=work/'train_s1_truth.complete.json'
    if marker.exists() and target.exists():return target,json.loads(marker.read_text())
    if target.exists():target.unlink()
    db=dbopen(target)
    db.execute('CREATE TABLE s1(id TEXT PRIMARY KEY,name TEXT,addr TEXT,country TEXT)')
    db.execute('CREATE TABLE truth(id TEXT PRIMARY KEY,matches TEXT)')
    count=0
    for part in chunks(stream_source(root/'dataset/train/train_source1.tsv'),15000):
        db.executemany('INSERT INTO s1 VALUES(?,?,?,?)',part)
        count+=len(part);db.commit()
        if count%250000<15000:LOG.info('Loaded train S1: %d',count)
    ntruth=0;total=0
    with (root/'dataset/train/train_ground_truth.tsv').open(encoding='utf8',newline='') as f:
        reader=csv.DictReader(f,delimiter='\t')
        def values():
            nonlocal ntruth,total
            for row in reader:
                links=list(dict.fromkeys(filter(None,row['matched_entity_ids'].split(','))))
                ntruth+=1;total+=len(links)
                yield (row['source1_entity_id'],','.join(links))
        for part in chunks(values(),15000):
            db.executemany('INSERT INTO truth VALUES(?,?)',part);db.commit()
    joined=db.execute('SELECT count(*) FROM s1 JOIN truth ON s1.id=truth.id').fetchone()[0]
    db.close()
    if joined!=count or ntruth!=count:raise RuntimeError(f'Train S1/truth misalignment: {count}/{ntruth}/{joined}')
    state={'s1':count,'positive_links':total};atomic(marker,state)
    LOG.info('FULL GT indexed: %s',state)
    return target,state

def holdout(s1id,bps):
    return int.from_bytes(hashlib.blake2b(s1id.encode(),digest_size=4,person=b'V13split').digest(),'little')%10000<bps

def sql_hard(db,r,correct,cap):
    out=[]
    for field,variants,minlen in (('name_key',[nkey(r[1]),indexed_nkey(r[1])],5),
                                  ('addr_key',[akey(r[2])],9)):
        vals=list(dict.fromkeys(v for v in variants if len(v)>=minlen))
        if not vals:continue
        q=f'SELECT id,name,addr,country FROM targets WHERE country=? AND {field} IN ('+','.join('?'*len(vals))+') LIMIT ?'
        out.extend(tuple(x) for x in db.execute(q,(r[3],*vals,cap+1)) if x[0] not in correct)
    return list({x[0]:x for x in out}.values())[:cap]

def train_batch(rows,db,comp,rng,a):
    # rows: (s1 normalized tuple, positive IDs) ; consume every positive link.
    allids=[mid for _,ids in rows for mid in ids]
    targets=get_targets(db,allids)
    if len(targets)!=len(set(allids)):
        missing=set(allids)-set(targets)
        raise RuntimeError(f'Training target IDs absent from TRAIN S2/S3 DB: {list(missing)[:5]}')
    records_by_country=defaultdict(list);by_token=defaultdict(list);by_number=defaultdict(list)
    for _,ids in rows:
        for mid in ids:
            t=targets[mid];records_by_country[t[3]].append(t)
            words=nkey(t[1]).split()
            if words:by_token[(t[3],words[0])].append(t)
            nums=numbers(t[2])
            if nums:by_number[(t[3],sorted(nums)[0])].append(t)
    pairs=[];labels=[]
    for r,true_ids in rows:
        good=set(true_ids)
        for mid in true_ids:
            pairs.append((r,targets[mid]));labels.append(1)
        possible=[]
        words=nkey(r[1]).split()
        if words:possible.extend(by_token.get((r[3],words[0]),[]))
        nums=numbers(r[2]);
        if nums:possible.extend(by_number.get((r[3],sorted(nums)[0]),[]))
        rng.shuffle(possible)
        seen=set(good);neg=[]
        for t in possible:
            if t[0] not in seen:
                seen.add(t[0]);neg.append(t)
                if len(neg)>=a.neg_per_entity:break
        # Fallback to in-batch negatives from the SAME country.
        pool=records_by_country.get(r[3],[])
        for _ in range(min(30,len(pool)*2)):
            if len(neg)>=a.neg_per_entity:break
            t=pool[rng.randrange(len(pool))]
            if t[0] not in seen:seen.add(t[0]);neg.append(t)
        if a.sql_hard_every and rng.randrange(a.sql_hard_every)==0:
            neg.extend(t for t in sql_hard(db,r,good,a.sql_hard_cap) if t[0] not in seen)
        for t in neg[:a.neg_per_entity+a.sql_hard_cap]:
            pairs.append((r,t));labels.append(0)
    return comp.compute(pairs),np.asarray(labels,np.uint8)

def fit_lgb(X,y,threads,iterations):
    LOG.info('Fitting new LightGBM on %d full-data supervised pairs; positive=%d',len(y),int(y.sum()))
    weights=np.where(y==0,5.0,1.0).astype(np.float32)
    clf=lgb.LGBMClassifier(n_estimators=iterations,num_leaves=47,max_depth=-1,
        learning_rate=.065,min_child_samples=180,colsample_bytree=.92,
        reg_lambda=6.,verbosity=-1,n_jobs=threads,random_state=202613,
        max_bin=127)
    clf.fit(X,y,sample_weight=weights)
    return clf

def make_training(a,root,work):
    datafile,state=ensure_train_sources(root,work)
    targetfile=ensure_targets(root,work,'train')
    cap=state['positive_links']+state['s1']*(a.neg_per_entity+a.sql_hard_cap)
    featurefile=work/'full_features.f32';labelfile=work/'full_labels.u8';meta=work/'training_complete.json'
    if meta.exists():
        summary=json.loads(meta.read_text());
        if summary.get('schema')!=FCOUNT or summary.get('params')!=[a.holdout_bps,a.neg_per_entity,a.sql_hard_every,a.sql_hard_cap]:
            raise RuntimeError('Training cache uses different settings; use a new --work')
        LOG.info('Reusing full training features: %s',summary)
        X=np.memmap(featurefile,dtype=np.float32,mode='r',shape=(summary['capacity'],FCOUNT))
        y=np.memmap(labelfile,dtype=np.uint8,mode='r',shape=(summary['capacity'],))
        return X[:summary['written']],y[:summary['written']],summary['holdout']
    LOG.info('Generating FULL training features: capacity=%d rows (%.2f GB)',cap,cap*FCOUNT*4/1e9)
    X=np.memmap(featurefile,dtype=np.float32,mode='w+',shape=(cap,FCOUNT))
    y=np.memmap(labelfile,dtype=np.uint8,mode='w+',shape=(cap,))
    db=dbopen(datafile,True);targets=dbopen(targetfile,True)
    rng=random.Random(202613);comp=FeatureComputer(a.workers)
    hold=[];used=0;entities=0;seenlinks=0
    try:
        cursor=db.execute('SELECT s.id,s.name,s.addr,s.country,t.matches FROM s1 s JOIN truth t ON s.id=t.id ORDER BY s.rowid')
        for chunk in chunks(cursor,a.train_batch):
            tr=[]
            for rid,name,addr,country,cell in chunk:
                matches=tuple(filter(None,cell.split(',')))
                if holdout(rid,a.holdout_bps):
                    hold.append(((rid,name,addr,country),matches))
                    continue
                tr.append(((rid,name,addr,country),matches));seenlinks+=len(matches)
            if not tr:continue
            xx,yy=train_batch(tr,targets,comp,rng,a)
            stop=used+len(yy)
            if stop>cap:raise RuntimeError('Feature allocation exceeded; re-run with a larger capacity')
            X[used:stop]=xx;y[used:stop]=yy;used=stop;entities+=len(tr)
            if entities%50000<a.train_batch:
                X.flush();y.flush();LOG.info('FULL training: %d/%d S1, %d pairs, %d positives',entities,state['s1'],used,seenlinks)
        X.flush();y.flush()
    finally:
        comp.close();db.close();targets.close()
    # Store holdout records for the independent threshold calibration; no GT in test.
    holdpath=work/'fresh_holdout.json'
    atomic(holdpath,{'rows':hold,'bps':a.holdout_bps})
    summary={'capacity':cap,'written':used,'entities_trained':entities,'positive_links_used':seenlinks,
             'holdout':str(holdpath),'schema':FCOUNT,
             'params':[a.holdout_bps,a.neg_per_entity,a.sql_hard_every,a.sql_hard_cap]}
    atomic(meta,summary)
    LOG.info('Training features complete: %s',summary)
    return X[:used],y[:used],str(holdpath)

_RET_DB=None
_RET_DF=None
_RET_FTS=False

def retrieval_worker_init(dbpath,enable_fts):
    global _RET_DB,_RET_DF,_RET_FTS
    _RET_DB=dbopen(dbpath,True)
    _RET_FTS=enable_fts and bool(_RET_DB.execute(
        "SELECT 1 FROM sqlite_master WHERE name='targets_fts'").fetchone())
    _RET_DF={}
    if _RET_FTS:
        _RET_DB.execute('PRAGMA query_only=OFF')
        try:
            _RET_DB.execute("CREATE VIRTUAL TABLE temp.v13vocab USING fts5vocab(main,targets_fts,'row')")
        finally:_RET_DB.execute('PRAGMA query_only=ON')
        _RET_DF=dict(_RET_DB.execute('SELECT term,doc FROM v13vocab'))

def retrieval_worker_job(payload):
    from types import SimpleNamespace
    rows,exact_cap,fts_cap=payload
    obj=Hybrid.__new__(Hybrid)
    obj.db=_RET_DB;obj.df=_RET_DF;obj.fts=_RET_FTS
    obj.a=SimpleNamespace(exact_cap=exact_cap,fts_cap=fts_cap,fts=_RET_FTS)
    return [(obj.exact(r),obj.fts_hits(r)) for r in rows]

class Hybrid:
    """New exact + rare-token SQLite FTS retrieval; optional cached 3-channel ANN."""
    def __init__(self,root,work,split,a):
        self.a=a;self.root=root;self.split=split;self.db=dbopen(ensure_targets(root,work,split),True)
        self.fts=bool(self.db.execute("SELECT 1 FROM sqlite_master WHERE name='targets_fts'").fetchone())
        self.df={};self.ann=None;self.country=None;self.ret_pool=None
        self.folder=root/('artifacts_v7_fresh' if split=='train' else 'artifacts_v8_test')
        if a.ann:
            try:
                import faiss,torch,joblib,model_v4_hybrid as v4,model_v7_fresh as v7,__main__
                __main__.FieldEncoder=v7.FieldEncoder;__main__.Encoder=v4.Encoder
                self.faiss=faiss;self.joblib=joblib
                self.enc={field:v7.load_encoder(root/'artifacts_v7_fresh'/f'encoder_{field}_d64_s60000.joblib',torch.device('cpu')) for field in ('combined','name','addr')}
                self.ann={};self.faiss.omp_set_num_threads(a.threads)
                LOG.info('Loaded existing encoders; ANN expansion enabled')
            except (ImportError,FileNotFoundError) as exc:
                LOG.warning('ANN unavailable (%s); SQL/FTS retrieval remains active',exc)
                self.ann=None
        if a.retrieval_workers:
            self.ret_pool=ProcessPoolExecutor(max_workers=a.retrieval_workers,
                initializer=retrieval_worker_init,
                initargs=(str(ensure_targets(root,work,split)),a.fts))
        if self.fts and a.fts and not self.ret_pool:
            # A temp fts5vocab is writable even with the main database opened mode=ro;
            # query_only must be OFF while creating the temp schema object.
            self.db.execute('PRAGMA query_only=OFF')
            try:
                self.db.execute("CREATE VIRTUAL TABLE temp.v13vocab USING fts5vocab(main,targets_fts,'row')")
            finally:
                self.db.execute('PRAGMA query_only=ON')
            self.df=dict(self.db.execute('SELECT term,doc FROM v13vocab'))
    def activate(self,country):
        if self.ann is None or self.country==country:return
        self.ann.clear();gc.collect();self.country=country
        for s in (2,3):
            for field in ('combined','name','addr'):
                folder=self.folder/('combined_indexes' if field=='combined' else 'field_indexes')
                tag=f'{s}_{country}';base=f'{tag}.faiss' if field=='combined' else f'{field}_{tag}.faiss'
                path=folder/base;idsfile=folder/f'{tag}.ids.joblib'
                if not path.exists() or not idsfile.exists():
                    LOG.warning('ANN index unavailable: %s; SQL/FTS fallback',path);self.ann=None;return
                ix=self.faiss.read_index(str(path));ix.hnsw.efSearch=self.a.ef
                ids=self.joblib.load(idsfile)
                if ix.ntotal!=len(ids):raise RuntimeError(f'ANN index/IDs inconsistent: {path}')
                self.ann[(s,field)]=(ix,ids)
    def exact(self,r):
        found={}
        for key,variants,minimum in (('name_key',[nkey(r[1]),indexed_nkey(r[1])],4),
                                     ('addr_key',[akey(r[2])],8)):
            vals=list(dict.fromkeys(v for v in variants if len(v)>=minimum))
            if not vals:continue
            sql=f'SELECT id,name,addr,country FROM targets WHERE country=? AND {key} IN ('+','.join('?'*len(vals))+') LIMIT ?'
            for b in self.db.execute(sql,(r[3],*vals,self.a.exact_cap)):
                found[b[0]]=tuple(b)
        return found
    def fts_hits(self,r):
        if not self.fts or not self.a.fts:return {}
        result={};sql=('SELECT t.id,t.name,t.addr,t.country FROM targets_fts '
            'JOIN targets t ON t.rowid=targets_fts.rowid WHERE targets_fts MATCH ? AND t.country=? LIMIT ?')
        for field,value in (('name',nkey(r[1])),('addr',akey(r[2]))):
            tokenset=set(x for x in value.split() if len(x)>=3 and x not in STOP_TOKENS and x in self.df)
            ranked=sorted(tokenset,key=lambda t:self.df[t])[:3]
            if len(ranked)>=2:
                terms=ranked[:2];probes=[' AND '.join(f'{field}:"{x}"' for x in terms)]
                if len(ranked)==3:probes.append(' AND '.join(f'{field}:"{x}"' for x in (ranked[0],ranked[2])))
            elif ranked and self.df[ranked[0]]<=self.a.fts_cap:
                probes=[f'{field}:"{ranked[0]}"']
            else:probes=[]
            for probe in probes[:2]:
                for b in self.db.execute(sql,(probe,r[3],self.a.fts_cap)):
                    result[b[0]]=tuple(b)
        return result
    def ann_batch(self,rows):
        if self.ann is None or not rows:return [{} for _ in rows]
        self.activate(rows[0][3])
        if self.ann is None:return [{} for _ in rows]
        out=[{} for _ in rows]
        for field in ('combined','name','addr'):
            vec=np.ascontiguousarray(self.enc[field].transform(rows,batch_size=min(256,len(rows))),np.float32)
            for s in (2,3):
                ix,ids=self.ann[(s,field)];_,labels=ix.search(vec,min(self.a.k,ix.ntotal))
                for d,hits in zip(out,labels):
                    for ixid in hits:
                        if ixid>=0:d[ids[int(ixid)]]=None
        return out
    def retrieve(self,rows,baselines=None,old_candidates=None):
        if not rows:return []
        if any(r[3]!=rows[0][3] for r in rows):raise ValueError('Country mixed batch')
        # Wider ANN is used on uncertain rows (and on ALL fresh validation rows).
        asks=[i for i,r in enumerate(rows) if baselines is None or self.a.expand_all or
              len(baselines[i])<self.a.expand_when_fewer_than or
              not ({k[:2] for k in baselines[i]}=={'S2','S3'})]
        lex=[{} for _ in rows]
        tasks=[]
        if asks and self.ret_pool:
            for positions in chunks(asks,24):
                future=self.ret_pool.submit(retrieval_worker_job,
                    ([rows[i] for i in positions],self.a.exact_cap,self.a.fts_cap))
                tasks.append((positions,future))
        ANN=[{} for _ in rows]
        if asks and self.ann is not None:
            for part in chunks(asks,120):
                hits=self.ann_batch([rows[i] for i in part])
                for i,h in zip(part,hits):ANN[i]=h
        if tasks:
            for positions,future in tasks:
                for idx,(ex,fts) in zip(positions,future.result()):
                    lex[idx].update(ex);lex[idx].update(fts)
        elif not self.ret_pool:
            for i in asks:
                lex[i].update(self.exact(rows[i]))
                lex[i].update(self.fts_hits(rows[i]))
        result=[]
        for i,r in enumerate(rows):
            old=old_candidates[i] if old_candidates else []
            required=list(dict.fromkeys((baselines[i] if baselines else [])+old[:self.a.old_cap]))
            found={cid:None for cid in required}
            if i in asks:
                found.update(lex[i]);found.update(ANN[i])
            result.append((found,required))
        return result
    def close(self):
        if self.ret_pool:self.ret_pool.shutdown()
        self.db.close();self.ann=None

def metric(pred,truth):
    if not truth:return float(not pred)
    return 1.25*len(pred&truth)/(.25*len(truth)+len(pred))

def classify_batch(clf,retriever,rows,comp,baselines=None,old_candidates=None,threshold=.8):
    candidates=retriever.retrieve(rows,baselines,old_candidates)
    # Get only metadata absent from exact/FTS retrieval; prevents N per-ID queries.
    needed=[cid for found,_ in candidates for cid,value in found.items() if value is None]
    resolved=get_targets(retriever.db,needed)
    pairs=[];ranges=[];keys=[]
    for r,(found,required) in zip(rows,candidates):
        for cid in list(found):
            if found[cid] is None:
                if cid in resolved:found[cid]=resolved[cid]
                else:del found[cid]
        # Every existing prediction is retained in ranking; filter remaining with
        # cheap fuzzy score before expensive feature extraction.
        original=set(required)
        def rank(b):
            n=fuzz.ratio(nkey(r[1]),nkey(b[1]))
            ad=fuzz.ratio(akey(r[2]),akey(b[2])) if b[2] else 0
            return max(n,ad)+.27*min(n,ad)
        chosen=list(dict.fromkeys([c for c in required if c in found] +
                [b[0] for b in sorted((v for k,v in found.items() if k not in original),
                                     key=rank,reverse=True)[:retriever.a.extra_cap + (retriever.a.old_cap if baselines is None else 0)]]))
        start=len(keys)
        for cid in chosen:
            pairs.append((r,found[cid]));keys.append(cid)
        ranges.append((start,len(keys)))
    x=comp.compute(pairs)
    probabilities=clf.predict_proba(x)[:,1] if len(x) else np.empty(0,np.float32)
    result=[]
    for (lo,hi),(found,required) in zip(ranges,candidates):
        mids=keys[lo:hi];prob=probabilities[lo:hi]
        result.append((mids,prob,required))
    return result

def validate(a,root,work,clf):
    hp=work/'fresh_holdout.json'
    data=json.loads(hp.read_text());entries=data['rows']
    if len(entries)<10:raise RuntimeError('Insufficient fresh held-out S1 records for validation')
    truth={r[0]:set(links) for r,links in entries}
    shuffled=list(entries);random.Random(202616).shuffle(shuffled)
    sample=shuffled[:min(a.valid_rows,a.holdout_rows)];comp=FeatureComputer(a.workers)
    retriever=Hybrid(root,work,'train',a)
    pred={};ceil=[];timed=time.monotonic()
    try:
        for country in sorted({r[3] for r,_ in sample}):
            items=[r for r,links in sample if r[3]==country]
            for batch in chunks(items,a.batch):
                result=classify_batch(clf,retriever,batch,comp,threshold=.8)
                for r,(ids,probs,_) in zip(batch,result):pred[r[0]]=(ids,probs.tolist())
    finally:retriever.close();comp.close()
    rng=random.Random(202614);ids=list(pred);rng.shuffle(ids)
    tune=ids[:len(ids)//2];report=ids[len(ids)//2:]
    if not tune or not report:raise RuntimeError('Too few validation rows')
    def score(group,t):
        return float(np.mean([metric({cid for cid,p in zip(*pred[rid]) if p>=t},truth[rid]) for rid in group]))
    grid=sorted(set(np.r_[np.arange(.20,.981,.025),np.arange(.98,1.,.0025),.9995].tolist()))
    threshold=max(grid,key=lambda t:score(tune,float(t)))
    recall=sum(len(set(pred[rid][0])&truth[rid]) for rid in report)/max(1,sum(len(truth[rid]) for rid in report))
    oracle=float(np.mean([metric(set(pred[rid][0])&truth[rid],truth[rid]) for rid in report]))
    summary={'threshold':float(threshold),'tuning_macro_f05':score(tune,threshold),
             'heldout_macro_f05':score(report,threshold),'pair_recall':recall,
             'candidate_oracle_macro_f05':oracle,'validation_entities':len(ids),
             'elapsed_s':time.monotonic()-timed,
             'note':'Fresh S1 hash holdout; target encoders are preexisting unsupervised. France has no labeled validation.'}
    atomic(work/'validation.json',summary);LOG.info('FRESH VALIDATION %s',json.dumps(summary))
    return summary

def train(a,root,work):
    X,y,_=make_training(a,root,work)
    clf=fit_lgb(X,y,a.threads,a.iterations)
    clf.booster_.save_model(str(work/'pre_validation_model.txt'))
    summary=validate(a,root,work,clf)
    if a.refit_all:
        # Include previously held-out GT positives and newly constructed negatives
        # in the final fit, AFTER evaluating/tuning on the independently heldout S1.
        hold=json.loads((work/'fresh_holdout.json').read_text())['rows']
        tdb=dbopen(ensure_targets(root,work,'train'),True)
        comp=FeatureComputer(a.workers);rng=random.Random(202615)
        blocks=[];ys=[]
        try:
            for part in chunks([(tuple(r),tuple(links)) for r,links in hold],a.train_batch):
                xx,yy=train_batch(part,tdb,comp,rng,a);blocks.append(xx);ys.append(yy)
        finally:tdb.close();comp.close()
        if blocks:
            # Training on ALL ground truth S1 would require refitting with the
            # whole withheld set; train with both corpora as a single LightGBM fit.
            hX=np.concatenate(blocks);hy=np.concatenate(ys)
            LOG.info('FINAL refit includes %d extra holdout pairs',len(hy))
            mat=np.concatenate((np.asarray(X),hX));lab=np.concatenate((np.asarray(y),hy))
            del blocks,ys,hX,hy,clf;gc.collect()
            clf=fit_lgb(mat,lab,a.threads,a.iterations)
            del mat,lab;gc.collect()
    clf.booster_.save_model(str(work/'final_model.txt'))
    atomic(work/'model_info.json',{'feature_count':FCOUNT,'threshold':summary['threshold'],
        'refit_all':a.refit_all,'fresh_validation_pre_refit':summary,
        'warning':'A refit changes calibration slightly; score must be verified on hidden test.'})
    LOG.info('FULL TRAINING COMPLETE: %s',work/'final_model.txt')

def ensure_test_s1(root,work):
    path=work/'test_s1.sqlite';marker=work/'test_s1.complete'
    if marker.exists() and path.exists():return path
    if path.exists():path.unlink()
    db=dbopen(path)
    db.execute('CREATE TABLE s1(id TEXT PRIMARY KEY,name TEXT,addr TEXT,country TEXT)')
    count=0
    for part in chunks(stream_source(root/'dataset/test/test_source1.tsv'),15000):
        db.executemany('INSERT INTO s1 VALUES(?,?,?,?)',part)
        count+=len(part);db.commit()
    db.close();marker.write_text(str(count));LOG.info('Indexed TEST S1: %d',count)
    return path

def load_baseline(root,a):
    m=a.baseline/'matching_results.tsv';c=a.baseline/'candidate_pairs.tsv'
    if not (m.exists() and c.exists()):
        raise FileNotFoundError('Full V10 matching_results.tsv AND candidate_pairs.tsv are required as retrieval seed')
    return m,c

def inference(a,root,work,out):
    import lightgbm as lgb
    metadata=json.loads((work/'model_info.json').read_text())
    if metadata['feature_count']!=FCOUNT:raise RuntimeError('Model feature schema mismatch')
    threshold=float(metadata['threshold'])
    clf=lgb.Booster(model_file=str(work/'final_model.txt'))
    # simple adapter so same code handles LGBMClassifier and raw Booster.
    class Adapter:
        def predict_proba(self,X):
            p=clf.predict(X,num_threads=a.threads)
            return np.stack((1-p,p),axis=1)
    base_m,base_c=load_baseline(root,a)
    s1file=ensure_test_s1(root,work);s1db=dbopen(s1file,True)
    retriever=Hybrid(root,work,'test',a);comp=FeatureComputer(a.workers)
    out.mkdir(parents=True,exist_ok=True)
    statefile=out/'resume.json';pm=out/'matching.partial';pc=out/'candidates.partial'
    settings={'baseline_matching':(str(base_m.resolve()),base_m.stat().st_size,base_m.stat().st_mtime_ns),
              'baseline_candidates':(str(base_c.resolve()),base_c.stat().st_size,base_c.stat().st_mtime_ns),
              'threshold':threshold,'k':a.k,'ef':a.ef,'ann':a.ann,'exact_cap':a.exact_cap,
              'fts':a.fts,'fts_cap':a.fts_cap,'old_cap':a.old_cap,'extra_cap':a.extra_cap,
              'expand_when_fewer_than':a.expand_when_fewer_than,'expand_all':a.expand_all,
              'model_sha256':hashlib.sha256((work/'final_model.txt').read_bytes()).hexdigest()}
    if statefile.exists():
        s=json.loads(statefile.read_text())
        if s['settings']!=settings:raise RuntimeError('Inference settings changed. Use a NEW --output directory')
    else:s={'settings':settings,'rows':0,'input_m':0,'input_c':0,'output_m':0,'output_c':0}
    n=s['rows'];began=time.monotonic()
    mf=pm.open('r+b' if pm.exists() else 'w+b');cf=pc.open('r+b' if pc.exists() else 'w+b')
    try:
        mf.truncate(s['output_m']);cf.truncate(s['output_c']);mf.seek(0,2);cf.seek(0,2)
        with base_m.open('rb') as bmi,base_c.open('rb') as bci:
            if not n:
                h1=bmi.readline().rstrip(b'\r\n');h2=bci.readline().rstrip(b'\r\n')
                if h1!=b'source1_entity_id\tmatched_entity_ids' or h2!=b'source1_entity_id\tcandidate_entity_ids':
                    raise RuntimeError('V10 baseline TSV headers incorrect')
            else:bmi.seek(s['input_m']);bci.seek(s['input_c'])
            done=False
            while not done:
                rows=[];bm=[];bc=[]
                for _ in range(a.batch):
                    m=bmi.readline();c=bci.readline()
                    if not m or not c:
                        if m!=c:raise RuntimeError('V10 matching/candidate row counts disagree')
                        done=True;break
                    mparts=m.decode('utf8').rstrip('\r\n').split('\t')
                    cparts=c.decode('utf8').rstrip('\r\n').split('\t')
                    if len(mparts)!=2 or len(cparts)!=2 or mparts[0]!=cparts[0]:
                        raise RuntimeError(f'V10 matching/candidate mismatch at output line {n+len(rows)+2}')
                    rows.append(mparts[0]);bm.append(list(dict.fromkeys(filter(None,mparts[1].split(',')))))
                    bc.append(list(dict.fromkeys(filter(None,cparts[1].split(',')))))
                if not rows:break
                targets={}
                for part in chunks(rows,450):
                    query='SELECT id,name,addr,country FROM s1 WHERE id IN ('+','.join('?'*len(part))+')'
                    targets.update({r[0]:tuple(r) for r in s1db.execute(query,part)})
                if len(targets)!=len(rows):raise RuntimeError('Unknown S1 IDs in V10 baseline: '+str(set(rows)-set(targets)))
                # Country-wise scoring reuses a single country's cached ANN indexes.
                scored=[None]*len(rows)
                groups=defaultdict(list)
                for i,rid in enumerate(rows):groups[targets[rid][3]].append(i)
                for country,positions in groups.items():
                    rr=[targets[rows[i]] for i in positions]
                    new=classify_batch(Adapter(),retriever,rr,comp,
                        [bm[i] for i in positions],[bc[i] for i in positions],threshold)
                    for i,val in zip(positions,new):scored[i]=val
                # Protect previously accepted V10 IDs from model distribution shift:
                # retain all V10 predictions unless new model very strongly rejects them,
                # and add NEW IDs only above an independently tuned threshold.
                for rid,baseline,original,(ids,probs,_) in zip(rows,bm,bc,scored):
                    scoredmap={k:float(p) for k,p in zip(ids,probs)}
                    # Pure new model; use --retain-baseline to make the change additive.
                    if a.retain_baseline:
                        selected=list(baseline)+[cid for cid in ids if cid not in baseline and scoredmap[cid]>=max(.995,threshold)]
                    else:
                        selected=[cid for cid in ids if scoredmap[cid]>=threshold]
                    # Exact V10 candidates plus all selected new IDs, preserving validator subset.
                    candidate=list(dict.fromkeys(original+selected))
                    mf.write((rid+'\t'+','.join(dict.fromkeys(selected))+'\n').encode('utf8'))
                    cf.write((rid+'\t'+','.join(candidate)+'\n').encode('utf8'))
                mf.flush();cf.flush();os.fsync(mf.fileno());os.fsync(cf.fileno())
                n+=len(rows)
                s={'settings':settings,'rows':n,'input_m':bmi.tell(),'input_c':bci.tell(),
                   'output_m':mf.tell(),'output_c':cf.tell()}
                atomic(statefile,s)
                if n%10000<a.batch:
                    LOG.info('TEST %d entities; %.1f rows/sec',n,n/max(.01,time.monotonic()-began))
    finally:
        mf.close();cf.close();retriever.close();comp.close();s1db.close()
    needed=int((root/'dataset/test/test_source1.tsv').exists())
    expected=int((work/'test_s1.complete').read_text()) if needed else None
    if expected is not None and n!=expected:raise RuntimeError(f'Incomplete S1 output: {n}/{expected}')
    for temp,final,header in ((pm,out/'matching_results.tsv',b'source1_entity_id\tmatched_entity_ids\n'),
                              (pc,out/'candidate_pairs.tsv',b'source1_entity_id\tcandidate_entity_ids\n')):
        build=final.with_name(final.name+'.building')
        with build.open('wb') as f:
            f.write(header)
            with temp.open('rb') as data:shutil.copyfileobj(data,f)
        os.replace(build,final)
    LOG.info('SUBMISSION READY: %s',out/'matching_results.tsv')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=('train','test','all'),default='all')
    ap.add_argument('--root',type=Path,default=Path('student_resource'))
    ap.add_argument('--work',type=Path,default=Path('student_resource/artifacts_v13'))
    ap.add_argument('--output',type=Path,default=Path('student_resource/output_v13'))
    ap.add_argument('--baseline',type=Path,default=Path('student_resource/output_v10'))
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--retrieval-workers',type=int,default=2)
    ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--train-batch',type=int,default=400)
    ap.add_argument('--holdout-bps',type=int,default=70,help='70 = 0.7%% fresh holdout before full-data refit')
    ap.add_argument('--holdout-rows',type=int,default=5000)
    ap.add_argument('--neg-per-entity',type=int,default=2)
    ap.add_argument('--sql-hard-every',type=int,default=12)
    ap.add_argument('--sql-hard-cap',type=int,default=2)
    ap.add_argument('--iterations',type=int,default=400)
    ap.add_argument('--valid-rows',type=int,default=4000)
    ap.add_argument('--refit-all',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--batch',type=int,default=300)
    ap.add_argument('--ann',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--k',type=int,default=16)
    ap.add_argument('--ef',type=int,default=80)
    ap.add_argument('--old-cap',type=int,default=24)
    ap.add_argument('--extra-cap',type=int,default=22)
    ap.add_argument('--exact-cap',type=int,default=35)
    ap.add_argument('--fts',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--fts-cap',type=int,default=32)
    ap.add_argument('--expand-when-fewer-than',type=int,default=3)
    ap.add_argument('--expand-all',action='store_true')
    ap.add_argument('--retain-baseline',action='store_true',help='Conservative: retain every V10 prediction; only add very high-confidence new IDs')
    a=ap.parse_args()
    if a.train_batch<=0 or a.workers<1 or a.retrieval_workers<0 or a.threads<1 or a.k<1 or not 0<a.holdout_bps<5000:
        ap.error('Invalid batch/workers/threads/k or --holdout-bps')
    a.work.mkdir(parents=True,exist_ok=True);a.output.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',stream=sys.stdout)
    if a.mode in ('all','train'):train(a,a.root,a.work)
    if a.mode in ('all','test'):inference(a,a.root,a.work,a.output)

if __name__=='__main__':main()
