#!/usr/bin/env python3
"""V8 deadline-oriented pipeline. Reuses completed V7 train indexes/candidates.
Train: retrains classifier with existing V7 candidates and compares against saved V7.
Test: builds TEST target indexes once and streams country-partitioned test S1.
No test labels or external business data are used. Accuracy/runtime are NOT guaranteed.
"""
from __future__ import annotations
import argparse, csv, gc, json, logging, os, shutil, hashlib
from collections import defaultdict
from pathlib import Path
from functools import lru_cache
import numpy as np
import joblib
import torch
import faiss
import xgboost as xgb
from sklearn.model_selection import train_test_split
import model_v4_hybrid as v4
import model_v5_experimental as v5
import model_v6_fast as v6
import model_v7_fresh as v7

LOG=logging.getLogger('v8')
def load_v7_encoders(v7work, args, device):
    """
    Load the original V7 encoders without retraining.

    V7 serialized FieldEncoder while running as __main__.
    Register the original class in the current __main__
    module before calling joblib.load().
    """
    import __main__

    # Required for compatibility with V7's existing joblib files.
    __main__.FieldEncoder = v7.FieldEncoder

    encoders = {}

    for field in ("combined", "name", "addr"):
        path = (
            v7work
            / f"encoder_{field}_d{args.dim}_s{args.svd_samples}.joblib"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing V7 encoder: {path}\n"
                "Do not retrain. Check the V7 artifact directory "
                "and the --dim / --svd-samples arguments."
            )

        LOG.info("Loading existing V7 %s encoder: %s", field, path)

        encoders[field] = v7.load_encoder(path, device)

    return encoders


def v7_split(data, sample):
    sampled=v4.sample_records(v4.source_path(data,'train',1),sample,seed=v7.SEED)
    train,valid=train_test_split(sampled,test_size=.2,random_state=v7.SEED)
    tune,report=train_test_split(valid,test_size=.5,random_state=v7.SEED+1)
    return train,valid,{r[0] for r in tune},{r[0] for r in report}

def read_cached_candidates(split, records, out):
    dirs=sorted(out.glob(f'{split}_candidate_chunks_*'))
    if len(dirs)!=1:
        raise RuntimeError(f'Expected exactly one V7 {split} candidate cache, found {len(dirs)}. Set --v7-cache explicitly.')
    return read_cached_from_dir(records,dirs[0])

def read_cached_from_dir(records, folder):
    by_country=defaultdict(list)
    for i,r in enumerate(records): by_country[r[3]].append(i)
    ans=[None]*len(records)
    # V7 chunks were saved at its original query-batch size; discover from chunk lengths.
    for country,positions in by_country.items():
        paths=sorted(folder.glob(f'{country}_*.joblib'))
        if not paths: raise FileNotFoundError(f'No V7 cached chunks for {country}: {folder}')
        offset=0
        for path in paths:
            rows=joblib.load(path)
            if offset+len(rows)>len(positions): raise RuntimeError('Cache longer than current split')
            for j,row in enumerate(rows): ans[positions[offset+j]]=row
            offset+=len(rows)
        if offset!=len(positions): raise RuntimeError(f'Incomplete/mismatched cache for {country}: {offset}/{len(positions)}')
    return ans

def score(records,candidates,store,model,batch):
    result={}
    for i in range(0,len(records),batch):
        result.update(v4.score_batch(records[i:i+batch],candidates[i:i+batch],store,model))
        if (i//batch)%5==0: LOG.info('Validation scoring %d/%d',min(i+batch,len(records)),len(records))
    return result

def tune_score(scored, truth, tune_ids, report_ids):
    tuning={k:v for k,v in scored.items() if k in tune_ids}
    grid=np.unique(np.r_[np.arange(.30,.981,.025),np.arange(.4,.971,.005)])
    best=max(((v5.evaluate(tuning,truth,float(t))[0],float(t)) for t in grid),key=lambda x:x[0])
    held,conf=v5.evaluate({k:v for k,v in scored.items() if k in report_ids},truth,best[1])
    whole,_=v5.evaluate(scored,truth,best[1])
    return {'tuning':best[0],'threshold':best[1],'heldout':held,'full_partly_tuned':whole,'heldout_tp_fp_fn':conf}

def train(args):
    root=args.data.parent; v7work=root/'artifacts_v7_fresh'; v7out=root/'output_v7'
    db=v7work/'train_targets.sqlite'
    if not db.exists(): raise FileNotFoundError(f'V7 database missing: {db}')
    out=root/'output_v8'; models=root/'models_v8'; out.mkdir(exist_ok=True);models.mkdir(exist_ok=True)
    train,valid,tune_ids,report_ids=v7_split(args.data,args.sample)
    truth=v7.selected_truth(args.data/'train'/'train_ground_truth.tsv',{r[0] for r in train+valid})
    LOG.info('Loading completed V7 candidate caches; NO FAISS loading or retrieval')
    if args.v7_cache:
        tr=read_cached_from_dir(train,Path(args.v7_cache)/'train')
        va=read_cached_from_dir(valid,Path(args.v7_cache)/'validation')
    else:
        tr=read_cached_candidates('train',train,v7out)
        va=read_cached_candidates('validation',valid,v7out)
    v4.features=v6.extra_features
    store=v4.TargetStore(db,64,cache_dir=None)
    v4.name_key=lru_cache(maxsize=32768)(v4.name_key)
    v4.addr_key=lru_cache(maxsize=32768)(v4.addr_key)
    LOG.info('Generating V8 training features from %d cached S1 candidates',len(train))
    X,y=v4.training_arrays(train,tr,truth,store,max_neg=args.max_neg)
    w=v6.positive_sample_weights(X,y,args.positive_weight)
    LOG.info('Training initial V8 XGBoost: %d pairs, %d positives',len(y),int(y.sum()))
    device='cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    initial=v6.fit_xgb(X,y,device,650,args.cpu_threads,sample_weight=w)
    LOG.info('Mining hard negatives from cached V7 candidates')
    xx,yy=v6.mine_negatives_fast(train,tr,truth,store,initial,args.mining_per_entity,args.mining_batch,args.feature_workers)
    LOG.info('Mined %d difficult negatives',len(yy))
    X=np.concatenate([X,xx]);y=np.concatenate([y,yy]);w=np.concatenate([w,np.ones(len(yy),np.float32)])
    del xx,yy,initial;gc.collect()
    LOG.info('Training V8 precision-oriented model')
    precision=v6.fit_xgb(X,y,device,850,args.cpu_threads,sample_weight=w,depth=7,child_weight=3)
    LOG.info('Training V8 recall-oriented model')
    recall=v6.fit_xgb(X,y,device,750,args.cpu_threads,sample_weight=w,depth=9,child_weight=2)
    del X,y,w,tr;gc.collect()
    LOG.info('Scoring validation for both models')
    s1=score(valid,va,store,precision,args.score_batch)
    s2=score(valid,va,store,recall,args.score_batch)
    # Both score dictionaries are generated from exactly the same ordered candidates.
    def blend(weight):
        result={}
        for rid,pairs in s1.items():
            other=dict(s2[rid]);result[rid]=[(cid,weight*p+(1-weight)*other[cid]) for cid,p in pairs]
        return result
    options={}
    for name,scored in [('precision',s1),('recall',s2),('blend_25',blend(.25)),('blend_50',blend(.5)),('blend_75',blend(.75))]:
        options[name]=tune_score(scored,truth,tune_ids,report_ids)
        LOG.info('Candidate %-10s tune %.5f heldout %.5f threshold %.3f',name,options[name]['tuning'],options[name]['heldout'],options[name]['threshold'])
    # Compare to existing V7, evaluated on exactly the SAME candidate pool and split.
    oldpath=root/'models_v7'/'v7_selected_xgboost.json'
    if oldpath.exists():
        old=xgb.XGBClassifier();old.load_model(str(oldpath));old.set_params(device='cpu',n_jobs=args.cpu_threads)
        old_scores=score(valid,va,store,old,args.score_batch)
        options['v7_original']=tune_score(old_scores,truth,tune_ids,report_ids)
        LOG.info('Existing V7 tune %.5f heldout %.5f',options['v7_original']['tuning'],options['v7_original']['heldout'])
        del old,old_scores
    chosen=max(options,key=lambda name:options[name]['tuning'])
    LOG.info('SELECTED %s using tuning subset ONLY',chosen)
    precision.save_model(str(models/'v8_precision.json'))
    recall.save_model(str(models/'v8_recall.json'))
    if chosen=='v7_original':
        shutil.copy2(oldpath,models/'v8_selected_v7.json')
    metadata={'selected':chosen,'selection':options[chosen], 'all_options':options,'feature_count':v6.FEATURE_COUNT,
              'sample':args.sample,'seed':v7.SEED,'test_retrieval':'three V7-style channels, optional FTS',
              'notes':'V7 validation split has been inspected in prior experiments; held-out is NOT pristine.'}
    (models/'v8_metadata.json').write_text(json.dumps(metadata,indent=2))
    (out/'v8_results.txt').write_text(json.dumps(metadata,indent=2))
    LOG.info('V8 HELD-OUT F0.5 %.5f; partly-tuned full %.5f',options[chosen]['heldout'],options[chosen]['full_partly_tuned'])
    LOG.info('Saved models and metadata. Training completed successfully.')
    store.close()

def build_test_indexes(args,work,encoders):
    # Build TEST S2/S3, never use TRAIN target indexes for test submissions.
    db=work/'train_targets.sqlite'; complete=work/'combined_indexes'/'complete.joblib'
    if not(db.exists() and complete.exists()):
        LOG.info('Building TEST combined FAISS and SQLite/FTS (one time)')
        store=v4.TargetStore(db,args.dim,m=args.hnsw_m,ef_search=args.ef_search,
                cache_dir=work/'combined_indexes',rebuild=True,max_candidates=args.base_cap,
                fts_per_probe=args.fts_per_probe)
        store.build(args.data,'test',encoders['combined'],chunk_size=args.index_batch)
        store.indexes.clear();store.ids.clear();store.close();del store;gc.collect()
    # V7 builder hardcodes split=train, so this test-specific variant is essential.
    folder=work/'field_indexes';folder.mkdir(parents=True,exist_ok=True)
    for source in (2,3):
        marker=folder/f'source_{source}.complete.joblib'
        if marker.exists():continue
        LOG.info('Building TEST name/address FAISS for S%d',source)
        indexes={};ids=defaultdict(list);count=0;buf=[]
        def ingest(chunk):
            nonlocal count
            embeddings={f:encoders[f].transform(chunk,batch_size=1500) for f in ('name','addr')}
            groups=defaultdict(list)
            for i,r in enumerate(chunk):groups[r[3]].append(i)
            for country,positions in groups.items():
                key=(source,country)
                for field in ('name','addr'):
                    k=(field,source,country)
                    if k not in indexes:
                        ix=faiss.IndexHNSWFlat(args.dim,args.hnsw_m,faiss.METRIC_INNER_PRODUCT)
                        ix.hnsw.efConstruction=args.ef_construction;ix.hnsw.efSearch=args.ef_search
                        indexes[k]=ix
                    indexes[k].add(np.ascontiguousarray(embeddings[field][positions],np.float32))
                ids[key].extend(chunk[i][0] for i in positions)
            count+=len(chunk)
            if count%200000<args.index_batch:LOG.info('TEST source %d field indexed %d',source,count)
        for rec in v4.iter_records(v4.source_path(args.data,'test',source)):
            buf.append(rec)
            if len(buf)>=args.index_batch:ingest(buf);buf=[]
        if buf:ingest(buf)
        for (src,country),values in ids.items():
            tag=f'{src}_{country}'
            for field in ('name','addr'):
                ix=indexes[(field,src,country)]
                if ix.ntotal!=len(values):raise RuntimeError('FAISS-ID alignment failed')
                faiss.write_index(ix,str(folder/f'{field}_{tag}.faiss'))
            joblib.dump(values,folder/f'{tag}.ids.joblib',compress=0)
        joblib.dump({'source':source,'keys':list(ids),'dim':args.dim},marker)
        del indexes,ids;gc.collect()
        LOG.info('TEST field indexes for S%d complete',source)
    return db

def partition_test_s1(args,work):
    folder=work/'s1_by_country';done=folder/'complete.json'
    if done.exists():return json.loads(done.read_text())
    folder.mkdir(parents=True,exist_ok=True)
    handles={};counts=defaultdict(int)
    try:
        for r in v4.iter_records(v4.source_path(args.data,'test',1)):
            country=r[3]
            if country not in handles:
                if not country.replace('_','').isalnum():raise ValueError(f'Unsafe country key: {country}')
                handles[country]=(folder/f'{country}.tsv').open('w',encoding='utf8',newline='')
            csv.writer(handles[country],delimiter='\t',lineterminator='\n').writerow(r)
            counts[country]+=1
    finally:
        for f in handles.values():f.close()
    tmp=done.with_suffix('.tmp');tmp.write_text(json.dumps(counts));tmp.replace(done)
    return dict(counts)

def test(args):
    root=args.data.parent; models=root/'models_v8';meta_path=models/'v8_metadata.json'
    if not meta_path.exists():raise FileNotFoundError('Run --mode train first to create V8 models')
    meta=json.loads(meta_path.read_text());chosen=meta['selected'];threshold=meta['selection']['threshold']
    if meta['feature_count']!=v6.FEATURE_COUNT:raise RuntimeError('Feature schema changed; do not infer')
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    v7work=root/'artifacts_v7_fresh';testwork=root/'artifacts_v8_test'
    testwork.mkdir(parents=True,exist_ok=True)
    # V7 encoder filenames encode dim/sample count. Do not silently fit new encoders.
   # Load the original V7 encoders.
    # This handles encoders serialized under __main__.FieldEncoder.
    encoders = load_v7_encoders(v7work, args, device)
    manifest=testwork/'test_index_settings.json'
    settings={'dim':args.dim,'svd_samples':args.svd_samples,'hnsw_m':args.hnsw_m,
              'sources':[(str(v4.source_path(args.data,'test',i)),v4.source_path(args.data,'test',i).stat().st_size,
                          v4.source_path(args.data,'test',i).stat().st_mtime_ns) for i in (1,2,3)]}
    if manifest.exists() and json.loads(manifest.read_text())!=settings:
        raise RuntimeError('TEST data or index configuration changed. Move/delete artifacts_v8_test and output_v8/test_country_progress before rerun.')
    if not manifest.exists():manifest.write_text(json.dumps(settings,indent=2))
    db=build_test_indexes(args,testwork,encoders)
    counts=partition_test_s1(args,testwork)
    LOG.info('Test S1 partitioned by country: %s',counts)
    v4.features=v6.extra_features
    # Load the exact model configuration selected during V8 training.
    model1 = xgb.XGBClassifier()
    model2 = None
    weight = 1.0

    if chosen == "v7_original":
        model_path = models / "v8_selected_v7.json"

    elif chosen == "precision":
        model_path = models / "v8_precision.json"

    elif chosen == "recall":
        model_path = models / "v8_recall.json"

    elif chosen.startswith("blend_"):
        model_path = models / "v8_precision.json"

        # Example: blend_50 means
        # 50% precision model + 50% recall model.
        weight = int(chosen.split("_")[1]) / 100.0

        if not 0.0 <= weight <= 1.0:
            raise ValueError(
                f"Invalid blending weight: {weight}"
            )

        recall_path = models / "v8_recall.json"

        if not recall_path.exists():
            raise FileNotFoundError(
                f"Missing recall model: {recall_path}"
            )

        model2 = xgb.XGBClassifier()
        model2.load_model(str(recall_path))
        model2.set_params(
            device="cpu",
            n_jobs=args.cpu_threads
        )

    else:
        raise ValueError(
            f"Unknown selected model in V8 metadata: {chosen}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing selected model: {model_path}"
        )

    model1.load_model(str(model_path))
    model1.set_params(
        device="cpu",
        n_jobs=args.cpu_threads
    )

    LOG.info(
        "Loaded V8 configuration: %s | "
        "primary=%s | blend_weight=%.2f",
        chosen,
        model_path.name,
        weight
    )
    out=root/'output_v8';out.mkdir(exist_ok=True)
    progress=out/'test_country_progress';progress.mkdir(exist_ok=True)
    retriever=v7.GroupRetriever(args,testwork,encoders,db)
    # For deadline mode, expensive per-entity alternate FTS can be disabled.
    retriever.store.extra_enabled=not args.fast_test
    if args.fast_test:
        retriever.store.fts_candidates=lambda records,per_probe=55: [[] for _ in records]
        LOG.warning('FAST TEST: disabling both standard and alternate FTS; recall may drop substantially versus V7 validation')
    for country in sorted(counts):
        destination=progress/f'{country}.tsv';candidates_path=progress/f'{country}.candidates.tsv'
        if destination.exists() and candidates_path.exists():
            LOG.info('Country %s already completed, skipping',country);continue
        # Resume within a country using an append-only checkpoint and idempotent atomic marker.
        tmp=progress/f'{country}.partial.tsv';tmpc=progress/f'{country}.candidates.partial.tsv'
        checkpoint=progress/f'{country}.checkpoint.json'
        completed=json.loads(checkpoint.read_text())['rows'] if checkpoint.exists() else 0
        mode='a' if completed else 'w'
        with tmp.open(mode,encoding='utf8',newline='') as fo,tmpc.open(mode,encoding='utf8',newline='') as fc:
            wo=csv.writer(fo,delimiter='\t',lineterminator='\n');wc=csv.writer(fc,delimiter='\t',lineterminator='\n')
            with (testwork/'s1_by_country'/f'{country}.tsv').open(encoding='utf8',newline='') as fi:
                reader=csv.reader(fi,delimiter='\t')
                for _ in range(completed):next(reader)
                while True:
                    raw=[]
                    for _ in range(args.query_batch):
                        try:raw.append(next(reader))
                        except StopIteration:break
                    if not raw:break
                    rows=[tuple(r) for r in raw]
                    candidates=retriever.retrieve(rows)
                    # Extract features ONCE for two-model inference.
                    targets=retriever.store.get_many([cid for c in candidates for cid,_ in c])
                    feats=[];ids=[];offsets=[]
                    for r,cs in zip(rows,candidates):
                        start=len(ids)
                        for cid,sim in cs:
                            if cid in targets:
                                feats.append(v6.extra_features(r,targets[cid],sim));ids.append(cid)
                        offsets.append((start,len(ids)))
                    if feats:
                        matrix=np.asarray(feats,np.float32)
                        p=model1.predict_proba(matrix)[:,1]
                        if model2:p=weight*p+(1-weight)*model2.predict_proba(matrix)[:,1]
                    else:p=[]
                    for r,cs,(lo,hi) in zip(rows,candidates,offsets):
                        matches=[cid for cid,prob in zip(ids[lo:hi],p[lo:hi]) if prob>=threshold]
                        wo.writerow([r[0],','.join(dict.fromkeys(matches))])
                        wc.writerow([r[0],','.join(dict.fromkeys(cid for cid,_ in cs))])
                    fo.flush();fc.flush();os.fsync(fo.fileno());os.fsync(fc.fileno())
                    completed+=len(rows)
                    checkpoint.write_text(json.dumps({'rows':completed}))
                    if completed%max(args.query_batch*10,1)<args.query_batch:
                        LOG.info('TEST %s %d/%d',country,completed,counts[country])
        if completed!=counts[country]:raise RuntimeError(f'Incomplete country {country}')
        tmp.replace(destination);tmpc.replace(candidates_path);checkpoint.unlink(missing_ok=True)
        LOG.info('TEST country %s completed',country)
    retriever.close()
    # Input S1 order is not needed; validator checks IDs and count.
    final=out/'matching_results.tsv';candidate=out/'candidate_pairs.tsv'
    for target,suffix,header in [(final,'.tsv',['source1_entity_id','matched_entity_ids']),
                                 (candidate,'.candidates.tsv',['source1_entity_id','candidate_entity_ids'])]:
        temp=target.with_suffix('.building')
        with temp.open('w',encoding='utf8',newline='') as dst:
            writer=csv.writer(dst,delimiter='\t',lineterminator='\n');writer.writerow(header)
            for country in sorted(counts):
                file=progress/(country+suffix)
                with file.open(encoding='utf8') as src:shutil.copyfileobj(src,dst)
        temp.replace(target)
    LOG.info('Submission files ready: %s and %s',final,candidate)

def main():
    p=argparse.ArgumentParser(description='V8 reuse V7 retrieval; train or full test submission')
    p.add_argument('--mode',choices=('train','test'),default='train')
    p.add_argument('--data',type=Path,default=Path('student_resource/dataset'))
    p.add_argument('--sample',type=int,default=10000)
    p.add_argument('--cpu-threads',type=int,default=8)
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--score-batch',type=int,default=75)
    p.add_argument('--mining-batch',type=int,default=60)
    p.add_argument('--feature-workers',type=int,default=2)
    p.add_argument('--mining-per-entity',type=int,default=10)
    p.add_argument('--max-neg',type=int,default=24)
    p.add_argument('--positive-weight',type=float,default=1.35)
    p.add_argument('--v7-cache',type=Path,help='Optional folder containing train/ and validation/ chunk directories')
    p.add_argument('--query-batch',type=int,default=100)
    p.add_argument('--dim',type=int,default=64)
    p.add_argument('--svd-samples',type=int,default=60000)
    p.add_argument('--top-k',type=int,default=40)
    p.add_argument('--field-top-k',type=int,default=40)
    p.add_argument('--hnsw-m',type=int,default=16)
    p.add_argument('--ef-construction',type=int,default=100)
    p.add_argument('--ef-search',type=int,default=192)
    p.add_argument('--index-batch',type=int,default=10000)
    p.add_argument('--base-cap',type=int,default=520)
    p.add_argument('--v5-cap',type=int,default=700)
    p.add_argument('--total-cap',type=int,default=950)
    p.add_argument('--exact-cap',type=int,default=70)
    p.add_argument('--fts-per-probe',type=int,default=55)
    p.add_argument('--extra-probe-results',type=int,default=65)
    p.add_argument('--fast-test',action='store_true',help='Disable slow alternate FTS; may reduce recall')
    args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    faiss.omp_set_num_threads(args.cpu_threads);torch.set_num_threads(args.cpu_threads)
    if args.mode=='train':train(args)
    else:test(args)
if __name__=='__main__':main()
