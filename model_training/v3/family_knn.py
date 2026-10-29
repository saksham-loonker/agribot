"""Distance-weighted neighbors with at most one vote per inferred family."""
import numpy as np
import torch


def family_knn_margin(training,labels,families,family_mass,query,k=5,device='cpu'):
    training=np.asarray(training,np.float32);query=np.asarray(query,np.float32);labels=np.asarray(labels);families=np.asarray(families);family_mass=np.asarray(family_mass,np.float64)
    if training.ndim!=2 or query.ndim!=2 or training.shape[1]!=query.shape[1] or len(labels)!=len(training) or len(families)!=len(training) or len(family_mass)!=len(training):raise ValueError('Neighbor data identities/shapes differ')
    if len(set(families.tolist()))<k or k<1:raise ValueError('Insufficient distinct training families')
    if not np.isfinite(training).all() or not np.isfinite(query).all() or not np.isin(labels,[0,1]).all() or not np.isfinite(family_mass).all() or (family_mass<=0).any():raise ValueError('Invalid neighbor evidence')
    bank=torch.from_numpy(training).to(device);square=bank.square().sum(1);output=[]
    with torch.inference_mode():
        for start in range(0,len(query),128):
            batch=torch.from_numpy(query[start:start+128]).to(device);distance=(batch.square().sum(1)[:,None]+square[None,:]-2*batch@bank.T).clamp_min(0)
            count=min(64,len(training));pending=list(range(len(batch)));scores=np.empty(len(batch),np.float64)
            while pending:
                values,indices=torch.topk(distance[pending],count,largest=False,sorted=True);values=values.cpu().numpy();indices=indices.cpu().numpy();retry=[]
                for position,(ds,ix) in enumerate(zip(values,indices)):
                    seen=set();chosen=[]
                    for d,i in zip(ds,ix):
                        if families[i] in seen:continue
                        seen.add(families[i]);chosen.append((float(d),int(i)))
                        if len(chosen)==k:break
                    if len(chosen)<k:retry.append(pending[position]);continue
                    weight=np.array([family_mass[i]/(d+1e-6) for d,i in chosen]);y=np.array([labels[i] for _,i in chosen]);positive=float(weight[y==1].sum());negative=float(weight[y==0].sum());scores[pending[position]]=np.log((positive+1e-12)/(negative+1e-12))
                if retry and count==len(training):raise ValueError('Missing distinct family evidence')
                pending=retry;count=min(len(training),count*2)
            output.extend(scores)
    return np.asarray(output,np.float64)
