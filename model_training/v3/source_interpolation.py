"""Within-label, cross-source training interpolation without validation access."""
import numpy as np

class CrossSourcePartners:
    def __init__(self,rows,fit):
        self.rows=rows;self.catalog={}
        for i in fit:
            i=int(i);r=rows[i];self.catalog.setdefault(r['label'],{}).setdefault(r['source'],{}).setdefault(r['group'],[]).append(i)

    def sample(self,selected,rng):
        partners=[];valid=[]
        for i in selected:
            r=self.rows[int(i)];pool={s:{g:ids for g,ids in groups.items() if g!=r['group']} for s,groups in self.catalog.get(r['label'],{}).items() if s!=r['source']};pool={s:g for s,g in pool.items() if g}
            if not pool:partners.append(int(i));valid.append(False);continue
            sources=sorted(pool);source=sources[int(rng.integers(len(sources)))];groups=sorted(pool[source]);group=groups[int(rng.integers(len(groups)))];choices=pool[source][group];partners.append(choices[int(rng.integers(len(choices)))]);valid.append(True)
        return np.array(partners),np.array(valid)

def partner_indices(rows,fit,selected,rng):
    return CrossSourcePartners(rows,fit).sample(selected,rng)
