"""V11 label-free specificity and reference-collision features."""
import csv, math, re, sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
import joblib
from rapidfuzz import fuzz
import model_v4_hybrid as v4
DF={}; STATS={'name':{},'addr':{}}

def build_stats(root,split,out):
    if out.exists():return
    counts={'name':Counter(),'addr':Counter()}
    with (root/'dataset'/split/f'{split}_source1.tsv').open(encoding='utf8',newline='') as f:
        for i,r in enumerate(csv.DictReader(f,delimiter='\t'),1):
            country=r['country'].strip().lower()
            for kind,key in [('name',v4.name_key(r['business_name'])),('addr',v4.addr_key(r['business_address']))]:
                if key:counts[kind][(country,key)]+=1
            if i%250000==0:print('Reference frequency count',split,i,flush=True)
    # Query records always belong to S1, so omitted nonempty keys have count one.
    stats={kind:{k:v for k,v in c.items() if v>1} for kind,c in counts.items()}
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix('.tmp');joblib.dump(stats,tmp,compress=0);tmp.replace(out)
    print('Reference duplicate keys',split,{k:len(v) for k,v in stats.items()},flush=True)

def configure(freq,stats):
    global DF,STATS
    DF=joblib.load(freq);STATS=joblib.load(stats)
    describe.cache_clear()

def sound(s):
    s=s.replace('ph','f').replace('sh','s').replace('ch','c').replace('ck','k').replace('qu','k').replace('v','w')
    return ' '.join(re.sub(r'(.)\1+',r'\1',re.sub('[aeiouhy]','',t)) for t in s.split())

@lru_cache(maxsize=24000)
def describe(name,addr,country):
    nk,ak=v4.name_key(name),v4.addr_key(addr)
    nt=set(re.findall('[a-z0-9]+',nk));at=set(re.findall('[a-z0-9]+',ak))
    nw={t:math.log1p(10000000/max(1,DF.get(t,1))) for t in nt}
    aw={t:math.log1p(10000000/max(1,DF.get(t,1))) for t in at}
    nc=STATS['name'].get((country,nk),1) if nk else 0
    ac=STATS['addr'].get((country,ak),1) if ak else 0
    return nk,ak,nw,aw,sound(nk),nc,ac

def overlaps(a,b):
    shared=sum(min(w,b[t]) for t,w in a.items() if t in b)
    sa,sb=sum(a.values()),sum(b.values())
    return [shared/max(1,sa),shared/max(1,sb),shared/max(1,sa+sb-shared),max((a[t] for t in a.keys()&b.keys()),default=0)]

def extra(a,b):
    n1,ad1,nw1,aw1,s1,nc,ac=describe(*a[1:])
    n2,ad2,nw2,aw2,s2,_,_=describe(*b[1:])
    return overlaps(nw1,nw2)+overlaps(aw1,aw2)+[
        math.log1p(nc),math.log1p(ac),float(nc==1),float(ac==1),
        max(nw1.values(),default=0),max(nw2.values(),default=0),
        max(aw1.values(),default=0),max(aw2.values(),default=0),
        fuzz.ratio(s1,s2)/100.,fuzz.token_sort_ratio(s1,s2)/100.,fuzz.WRatio(s1,s2)/100.,
        float(bool(n1) and n1==n2 and nc==1),float(bool(ad1) and ad1==ad2 and ac==1),
        float(bool(n1) and n1==n2 and nc>1),float(bool(ad1) and ad1==ad2 and ac>1),
        len(nw1),len(nw2),len(aw1),len(aw2)]

if __name__=='__main__':
    root=Path('student_resource');out=root/'output_v11'
    for split in ('train','test'):build_stats(root,split,out/f'source1_stats_{split}.joblib')
