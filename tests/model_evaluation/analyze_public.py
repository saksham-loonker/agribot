"""Auditable aggregation and notebook figures for the fixed-model investigation."""
from __future__ import annotations
import csv, hashlib, json, math, re, statistics, sys, xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from evaluate_public import LABELS, OTHER, OUT, ROOT, MANIFEST, posterior, replay

COLORS={'correct':'#3569A8','wrong':'#D28A26','other':'#A75B85','unsure':'#8B8D92','none':'#D6D7DA'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.labelcolor':'#24282C','text.color':'#24282C','figure.facecolor':'white','savefig.facecolor':'white'})


def wilson(k,n):
    if not n: return [None,None]
    p=k/n; z=1.96; den=1+z*z/n; center=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,center-half),min(1,center+half)]


def prediction(row,mode,bias=None,other_min_prob=.5):
    if mode=='whole':
        p=posterior(row['whole_member_logits'],bias)
        return LABELS[int(p.argmax())],float(p.max()),'CLASSIFIER'
    result=row['app_replay'] if bias is None and other_min_prob==.5 else replay(row['leaves'],row['quality_weight'],bias,other_min_prob)
    if result['kind'] in ('NO_LEAVES','UNSURE','NEED_MORE_VIEWS'): return result['kind'],result['confidence'],result['kind']
    return result['label'],result['confidence'],result['kind']


def scope(row):
    if row['label']!='Other': return 'supported_tomato' if row['label'] in LABELS else 'mixed_condition'
    if row['source']=='plantvillage': return 'unsupported_tomato'
    names=[obj['source_label'] for obj in row['oracle_objects']]
    if names and all(name.startswith('Tomato') for name in names): return 'unsupported_tomato'
    if names and all(not name.startswith('Tomato') for name in names): return 'non_tomato'
    return 'mixed_species_other'


def metrics(rows,mode='app',bias=None,other_min_prob=.5):
    selected=[r for r in rows if r['label'] in LABELS]
    triples=[(r,*prediction(r,mode,bias,other_min_prob)) for r in selected]
    known=[t for t in triples if t[0]['label']!='Other']; unknown=[t for t in triples if t[0]['label']=='Other']
    correct=sum(p==r['label'] for r,p,_,_ in triples)
    false_other=sum(p=='Other' for _,p,_,_ in known)
    false_known=sum(p in LABELS and p!='Other' for _,p,_,_ in unknown)
    rejected=sum(p=='Other' for _,p,_,_ in unknown)
    return dict(n=len(selected),known_n=len(known),unknown_n=len(unknown),correct=correct,
                accuracy=correct/len(selected) if selected else None,accuracy_ci95=wilson(correct,len(selected)),
                known_correct=sum(p==r['label'] for r,p,_,_ in known),known_false_other=false_other,
                known_false_other_rate=false_other/len(known) if known else None,known_false_other_ci95=wilson(false_other,len(known)),
                unknown_rejected=rejected,unknown_rejection_rate=rejected/len(unknown) if unknown else None,
                unknown_false_known=false_known,unknown_false_known_rate=false_known/len(unknown) if unknown else None,
                unsure=sum(p=='UNSURE' for _,p,_,_ in triples),no_leaves=sum(p=='NO_LEAVES' for _,p,_,_ in triples),
                known_unsure=sum(p=='UNSURE' for _,p,_,_ in known),known_no_leaves=sum(p=='NO_LEAVES' for _,p,_,_ in known),
                unknown_unsure=sum(p=='UNSURE' for _,p,_,_ in unknown),unknown_no_leaves=sum(p=='NO_LEAVES' for _,p,_,_ in unknown))


def save_csv(path,rows):
    if not rows: return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)


def test_counts():
    results={}
    for module in ('domain','camera','data','ml','featurescan','app'):
        total=failed=errors=skipped=0
        folder=ROOT/'agribot_android_app/android'/module/'build/test-results'
        for path in folder.glob('*/TEST-*.xml'):
            suite=ET.parse(path).getroot()
            total+=int(suite.get('tests',0)); failed+=int(suite.get('failures',0)); errors+=int(suite.get('errors',0)); skipped+=int(suite.get('skipped',0))
        results[module]=dict(tests=total,failures=failed,errors=errors,skipped=skipped)
    return results


def data_quality(rows):
    by_pixels=defaultdict(list)
    for row in rows: by_pixels[row['pixel_sha256']].append(row)
    conflicts=[{'ids':[r['id'] for r in group],'labels':sorted({r['label'] for r in group})} for group in by_pixels.values() if len({r['label'] for r in group})>1]
    conflict_hashes={h for h,group in by_pixels.items() if len({r['label'] for r in group})>1}
    unique=[group[0] for h,group in by_pixels.items() if h not in conflict_hashes]
    exact_unique_n=len(unique)
    tree=json.loads((OUT/'source/Tomato-Village_subtree0.json').read_text())['tree']
    training={e['sha'] for e in tree if e['type']=='blob' and e['path'].split('/')[0] in ('train','val')}
    testing={e['path']:e['sha'] for e in tree if e['type']=='blob' and e['path'].startswith('test/')}
    overlap=[path for path,sha in testing.items() if sha in training]
    pd_tree=json.loads((OUT/'source/PlantDoc-Object-Detection-Dataset_tree.json').read_text())['tree']
    pd_train={e['sha'] for e in pd_tree if e['path'].startswith('TRAIN/') and e['path'].lower().endswith('.jpg')}
    pd_overlap=[e['path'] for e in pd_tree if e['path'].startswith('TEST/') and e['path'].lower().endswith('.jpg') and e['sha'] in pd_train]
    excluded=[r['id'] for r in unique if (r['source']=='plantdoc' and r['source_path'] in pd_overlap) or
              (r['source']=='tomatovillage' and r['source_path'].split('/',1)[1] in overlap)]
    unique=[r for r in unique if r['id'] not in excluded]
    phash=[int(r['image_phash'],16) for r in unique]; parent=list(range(len(unique)))
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    pairs=0
    for i in range(len(unique)):
        for j in range(i):
            if (phash[i]^phash[j]).bit_count()<=4:
                pairs+=1; parent[find(i)]=find(j)
    clusters=defaultdict(list)
    for i,row in enumerate(unique): clusters[find(i)].append(row['id'])
    clustered=[group for group in clusters.values() if len(group)>1]
    keep_ids={group[0] for group in clusters.values()}
    near_unique=[r for r in unique if r['id'] in keep_ids]
    golden_dir=ROOT/'model_training/v2/bundle/golden'
    golden_hashes={hashlib.sha256(Image.open(p).convert('RGB').tobytes()).hexdigest() for p in golden_dir.glob('*.png')}
    return unique,near_unique,dict(downloaded=len(rows),exact_unique=exact_unique_n,analysis_n=len(unique),exact_duplicate_rows=len(rows)-len(by_pixels),
        conflicting_duplicate_groups=conflicts,near_duplicate_pairs_hamming4=pairs,near_duplicate_clusters=clustered,
        near_unique_n=len(near_unique),golden_pixel_overlap=sum(r['pixel_sha256'] in golden_hashes for r in rows),
        tomato_village_test_blob_overlap_with_train_or_val=overlap,plantdoc_test_blob_overlap_with_train=pd_overlap,excluded_training_overlap_ids=excluded,
        warning='Original full training image inventory/cache is unavailable; perceptual overlap with training and calibration cannot be ruled out. Public-source re-downloads are not evidence of unseen data.')


def main():
    rows=[json.loads(line) for line in (OUT/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    manifest=json.loads((OUT/'images_manifest.json').read_text(encoding='utf-8'))
    assert len(rows)==len(manifest),'Evaluation incomplete'
    assert len({r['id'] for r in rows})==len(rows),'Duplicate evaluation records'
    assert set(r['id'] for r in rows)==set(r['id'] for r in manifest),'Source/evaluation mismatch'
    unique,near_unique,dq=data_quality(rows)
    pv_groups=defaultdict(list)
    for row in unique:
        if row['source']!='plantvillage': continue
        identifier=Path(row['source_path']).stem.split('___')[-1].strip().lower()
        if 'leaf' in identifier: identifier=re.sub(r'(?<=\d)\.\d+$','',identifier)
        group=Path(row['source_path']).parent.name+':'+identifier
        pv_groups[group].append(row)
    representatives=[sorted(group,key=lambda r:r['id'])[0] for group in pv_groups.values()]
    dq['plantvillage_repeated_leaf_identifiers']=[dict(identifier=identifier,ids=[r['id'] for r in group]) for identifier,group in pv_groups.items() if len(group)>1]
    dq['plantvillage_leaf_identifier_sensitivity']=metrics(representatives)
    sources=sorted({r['source'] for r in unique}); summary=[]
    raw_summary=[dict(source=source,mode=mode,**metrics([r for r in rows if r['source']==source],mode)) for source in sources for mode in ('whole','app')]
    eligible_ids={r['id'] for r in unique}
    scope_metrics=[]
    for source in sources:
        for group in sorted({scope(r) for r in unique if r['source']==source}):
            selected=[r for r in unique if r['source']==source and scope(r)==group]
            for bias in [.75,0.]: scope_metrics.append(dict(source=source,scope=group,other_logit_bias=bias,**metrics(selected,bias=bias)))
    for source in sources:
        selected=[r for r in unique if r['source']==source]
        for mode in ('whole','app'):
            summary.append(dict(source=source,mode=mode,**metrics(selected,mode)))
    perclass=[]
    for source in sources:
        for label in LABELS:
            selected=[r for r in unique if r['source']==source and r['label']==label]
            if not selected: continue
            for mode in ('whole','app'): perclass.append(dict(source=source,label=label,mode=mode,**metrics(selected,mode)))
    for item in perclass:
        population=[r for r in unique if r['source']==item['source'] and r['label'] in LABELS]
        predictions=[(r['label'],prediction(r,item['mode'])[0]) for r in population]
        tp=sum(actual==item['label'] and predicted==item['label'] for actual,predicted in predictions)
        fp=sum(actual!=item['label'] and predicted==item['label'] for actual,predicted in predictions)
        fn=sum(actual==item['label'] and predicted!=item['label'] for actual,predicted in predictions)
        item.update(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
    common=sorted(set.intersection(*[{r['label'] for r in unique if r['source']==source and r['label'] not in ('Other','Mixed')} for source in sources]))
    common_class_comparison=[]
    for source in sources:
        for mode in ('whole','app'):
            selected=[r for r in perclass if r['source']==source and r['mode']==mode and r['label'] in common]
            common_class_comparison.append(dict(source=source,mode=mode,classes=common,n=sum(r['n'] for r in selected),
                macro_recall=float(np.mean([r['recall'] for r in selected])),per_class_n={r['label']:r['n'] for r in selected}))
    flat=[]
    for row in rows:
        app=row['app_replay']; p=np.asarray(row['whole_probabilities']); top=np.argsort(-p)
        flat.append(dict(id=row['id'],source=row['source'],label=row['label'],scope=scope(row),analysis_eligible=row['id'] in eligible_ids,whole_prediction=row['whole_label'],
                         whole_confidence=float(p.max()),whole_other_probability=float(p[OTHER]),whole_margin=float(p[top[0]]-p[top[1]]),
                         app_kind=app['kind'],app_label=app['label'],app_confidence=app['confidence'],
                         app_frames=app['frames'],detections=len(row['detections']),classified_leaves=len(row['leaves']),
                         app_partial=app['partial'],brightness=row['brightness'],sharpness=row['sharpness'],quality_accept=row['quality_accept'],runtime_threads=row.get('runtime_threads',4),
                         elapsed_host_ms=row['host_elapsed_ms'],path=row['path'],source_url=row['source_url'],pixel_sha256=row['pixel_sha256']))
    sensitivity=[]
    for bias in [-.5,0.,.25,.5,.75,1.]:
        for threshold in [.35,.5,.65]:
            for source in sources:
                sensitivity.append(dict(source=source,other_logit_bias=bias,other_min_probability=threshold,
                                        **metrics([r for r in unique if r['source']==source],bias=bias,other_min_prob=threshold)))
    oracle=[]
    for row in unique:
        for obj in row['oracle_objects']:
            oracle.append(dict(image_id=row['id'],source=row['source'],**obj))
    valid=[o for o in oracle if o['valid']]
    oracle_summary=[]
    for label in LABELS:
        objects=[o for o in valid if o['label']==label]
        if not objects: continue
        oracle_summary.append(dict(label=label,n=len(objects),correct=sum(o['prediction']==label for o in objects),
             false_other=sum(o['prediction']=='Other' for o in objects) if label!='Other' else None,
             detected_iou50=sum(o['best_detector_iou']>=.5 for o in objects),selected_iou50=sum(o['best_selected_iou']>=.5 for o in objects)))
    clustersensitivity=[dict(source=s,**metrics([r for r in near_unique if r['source']==s])) for s in sources]
    # Paired bias counterfactual: all else fixed; not a fitted/released threshold.
    paired={}
    for source in sources:
        known=[r for r in unique if r['source']==source and r['label'] not in ('Other','Mixed')]
        counts=Counter()
        for row in known:
            baseline=prediction(row,'app')[0]; no_bias=prediction(row,'app',bias=0.)[0]
            counts['baseline_unknown']+=baseline=='Other'
            counts['unknown_to_correct']+=baseline=='Other' and no_bias==row['label']
            counts['unknown_to_wrong_known']+=baseline=='Other' and no_bias in LABELS and no_bias not in ('Other',row['label'])
            counts['unknown_to_unsure']+=baseline=='Other' and no_bias=='UNSURE'
            counts['still_unknown']+=baseline=='Other' and no_bias=='Other'
        paired[source]=dict(known_n=len(known),**counts)
    quality_slices=[]
    for s in sources:
        for passed in (False,True):
            selected=[r for r in unique if r['source']==s and r['quality_accept']==passed]
            if selected: quality_slices.append(dict(source=s,quality_accept=passed,**metrics(selected)))
    output=dict(source_summary=summary,raw_source_summary=raw_summary,per_class=perclass,data_quality=dq,near_duplicate_sensitivity=clustersensitivity,
                oracle_summary=oracle_summary,invalid_oracle_boxes=sum(not o['valid'] for o in oracle),
                mixed_label_images=sum(r['label']=='Mixed' for r in rows),bias_paired=paired,quality_slices=quality_slices,
                tests=test_counts(),sample_counts=dict(Counter(r['source'] for r in rows)),common_class_comparison=common_class_comparison,scope_metrics=scope_metrics)
    calibration=[]; reliability=[]; bias_mechanism=[]
    for source in sources:
        selected=[r for r in unique if r['source']==source and r['label'] in LABELS]
        probs=np.asarray([r['whole_probabilities'] for r in selected]); truth=np.asarray([LABELS.index(r['label']) for r in selected])
        confidence=probs.max(axis=1); correct=probs.argmax(axis=1)==truth; ece=0.
        for low,high in zip(np.arange(0,1,.1),np.arange(.1,1.01,.1)):
            mask=(confidence>low)&(confidence<=high+1e-10)
            if not mask.any(): continue
            acc=float(correct[mask].mean()); conf=float(confidence[mask].mean()); ece+=mask.mean()*abs(acc-conf)
            reliability.append(dict(source=source,confidence_bin_low=round(low,1),confidence_bin_high=round(high,1),n=int(mask.sum()),accuracy=acc,mean_confidence=conf))
        target=np.eye(len(LABELS))[truth]
        calibration.append(dict(source=source,n=len(selected),ece10=float(ece),negative_log_likelihood=float(-np.log(np.maximum(probs[np.arange(len(truth)),truth],1e-12)).mean()),
                                multiclass_brier=float(np.square(probs-target).sum(axis=1).mean()),mean_confidence=float(confidence.mean()),accuracy=float(correct.mean()),
                                wrong_predictions_confident80=int(((~correct)&(confidence>=.8)).sum())))
        known=[r for r in selected if r['label']!='Other']
        whole_other=[r for r in known if r['whole_label']=='Other']
        bias_mechanism.append(dict(source=source,known_n=len(known),whole_false_other=len(whole_other),
                                  raw_model_already_other=int(sum(np.argmax(r['whole_member_logits'][0])==OTHER for r in whole_other)),
                                  bias_alone_flipped_to_other=int(sum(np.argmax(r['whole_member_logits'][0])!=OTHER for r in whole_other))))
    output['calibration']=calibration; output['whole_rejection_mechanism']=bias_mechanism
    save_csv(OUT/'calibration_metrics.csv',calibration); save_csv(OUT/'reliability_bins.csv',reliability)
    device_files=list((ROOT/'agribot_android_app/android/app/build/outputs/androidTest-results/connected').rglob('TEST-*.xml'))
    device=dict(tests=0,failures=0,errors=0,skipped=0)
    for path in device_files:
        suite=ET.parse(path).getroot()
        for field in device: device[field]+=int(suite.get(field,0))
    output['device_golden_tests']=device
    control_file=OUT/'synthetic_control_results.json'
    if control_file.exists():
        output['synthetic_controls']=[dict(name=r['id'],detections=len(r['detections']),whole_prediction=r['whole_label'],app_kind=r['app_replay']['kind'],quality_accept=r['quality_accept']) for r in json.loads(control_file.read_text())]
    stress_file=OUT/'stress_predictions.jsonl'
    if stress_file.exists():
        stress=[json.loads(line) for line in stress_file.read_text().splitlines()]
        output['stress']=[dict(transform=t,n=len(group),baseline_correct=sum(r['baseline']==r['label'] for r in group),
             transformed_correct=sum(r['prediction']==r['label'] for r in group),other_predictions=sum(r['prediction']=='Other' for r in group),
             changed=sum(r['prediction']!=r['baseline'] for r in group))
             for t in sorted({r['transform'] for r in stress}) for group in [[r for r in stress if r['transform']==t]]]
    (OUT/'analysis_summary.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    save_csv(OUT/'image_results.csv',flat); save_csv(OUT/'source_metrics.csv',summary); save_csv(OUT/'class_metrics.csv',perclass)
    save_csv(OUT/'bias_sensitivity.csv',sensitivity); save_csv(OUT/'oracle_object_results.csv',oracle)
    save_csv(OUT/'scope_metrics.csv',scope_metrics)
    save_csv(OUT/'oracle_class_metrics.csv',oracle_summary)
    figures=OUT/'figures'; figures.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(11,4.8)); labels=[]; components=defaultdict(list)
    for source in sources:
        m=next(r for r in summary if r['source']==source and r['mode']=='app'); n=m['known_n']; labels.append(f'{source}\n(n={n} known-condition images)')
        components['Correct known label'].append(m['known_correct']/n*100)
        components['Wrong known label'].append((n-m['known_correct']-m['known_false_other']-m['known_unsure']-m['known_no_leaves'])/n*100)
        components['Unknown condition'].append(m['known_false_other']/n*100)
        components['Unsure'].append(m['known_unsure']/n*100)
        components['No leaf detected'].append(m['known_no_leaves']/n*100)
    left=np.zeros(len(sources))
    for (name,values),color in zip(components.items(),COLORS.values()):
        ax.barh(np.arange(len(sources)),values,left=left,color=color,label=name,height=.58)
        for i,v in enumerate(values):
            if v>=5: ax.text(left[i]+v/2,i,f'{v:.1f}%',ha='center',va='center',color='white' if color not in (COLORS['none'],COLORS['unsure']) else '#24282C',fontsize=10)
        left+=values
    ax.set_yticks(np.arange(len(sources)),labels); ax.set_xlim(0,100); ax.set_xlabel('Share of known-condition images (%)')
    ax.set_title('Static app replay: outcomes on supported conditions',loc='left',pad=15)
    ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.2),frameon=False); fig.tight_layout(); fig.savefig(figures/'known_condition_outcomes.png',dpi=160,bbox_inches='tight'); plt.close(fig)
    tv=[r for r in perclass if r['source']=='tomatovillage' and r['mode']=='app']
    fig,ax=plt.subplots(figsize=(10,6)); positions=np.arange(len(tv)); rates=[r['accuracy']*100 for r in tv]
    intervals=np.array([r['accuracy_ci95'] for r in tv])*100
    ax.barh(positions,rates,color=COLORS['correct'],height=.6)
    ax.errorbar(rates,positions,xerr=np.vstack((np.array(rates)-intervals[:,0],intervals[:,1]-np.array(rates))),fmt='none',ecolor='#24282C',capsize=3)
    ax.set_yticks(positions,[f"{r['label']} (n={r['n']})" for r in tv]); ax.set_xlim(0,105); ax.set_xlabel('Correct app verdict (%) · nominal 95% Wilson intervals')
    ax.set_title('Tomato-Village test: all eight supported classes',loc='left',pad=15); fig.tight_layout(); fig.savefig(figures/'per_class_accuracy.png',dpi=160,bbox_inches='tight'); plt.close(fig)
    predicted=LABELS+['UNSURE','NO_LEAVES']; matrix=np.zeros((9,len(predicted)),int)
    for row in unique:
        if row['source']=='tomatovillage' and row['label'] in LABELS:
            pred=prediction(row,'app')[0]; matrix[LABELS.index(row['label']),predicted.index(pred)]+=1
    save_csv(OUT/'tomato_village_confusion.csv',[dict(label=LABELS[i],**{p:int(matrix[i,j]) for j,p in enumerate(predicted)}) for i in range(9)])
    fig,ax=plt.subplots(figsize=(13,7)); im=ax.imshow(matrix[:8],cmap='Blues',vmin=0)
    ax.set_yticks(range(8),LABELS[:8]); ax.set_xticks(range(len(predicted)),[s.replace(' Deficiency',' Def.').replace('Spotted Wilt Virus','Wilt virus') for s in predicted],rotation=45,ha='right')
    for (i,j),value in np.ndenumerate(matrix[:8]): ax.text(j,i,str(value),ha='center',va='center',color='white' if value>matrix.max()/2 else '#24282C',fontsize=10)
    ax.set_xlabel('Predicted app outcome'); ax.set_ylabel('Source label'); ax.set_title('Tomato-Village: app-replay confusion counts',loc='left',pad=15)
    fig.colorbar(im,ax=ax,label='Images'); fig.tight_layout(); fig.savefig(figures/'confusion_matrix.png',dpi=160,bbox_inches='tight'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for source,color in zip(sources,['#3569A8','#A75B85','#D28A26']):
        subset=[s for s in sensitivity if s['source']==source and s['other_min_probability']==.5]
        for ax,key in zip(axes,['known_false_other_rate','unknown_false_known_rate']):
            points=[s for s in subset if s[key] is not None]
            if points: ax.plot([s['other_logit_bias'] for s in points],[s[key]*100 for s in points],marker='o',label=source,color=color)
    for ax,title in zip(axes,['Supported condition → Unknown','Unsupported condition → Known label']):
        ax.axvline(.75,color='#444444',linestyle='--',linewidth=1,label='Shipped bias' if ax is axes[0] else None)
        ax.set_ylim(bottom=0); ax.set_xlabel('Other logit bias'); ax.set_ylabel('Images (%)'); ax.set_title(title,loc='left')
    axes[0].legend(frameon=False,fontsize=9); axes[1].legend(frameon=False,fontsize=9); fig.suptitle('Counterfactual bias sweep · model weights unchanged',x=.05,ha='left'); fig.tight_layout(); fig.savefig(figures/'bias_tradeoff.png',dpi=160,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6))
    ax.plot([0,1],[0,1],color='#444444',linestyle='--',label='Perfect calibration')
    for source,color in zip(sources,['#3569A8','#A75B85','#D28A26']):
        bins=[b for b in reliability if b['source']==source]
        ax.plot([b['mean_confidence'] for b in bins],[b['accuracy'] for b in bins],marker='o',color=color,label=source)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xlabel('Mean whole-image top-class confidence'); ax.set_ylabel('Observed source-label accuracy')
    ax.set_title('Whole-image reliability · equal-width bins',loc='left',pad=15); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(figures/'reliability.png',dpi=160,bbox_inches='tight'); plt.close(fig)
    examples=[r for r in unique if r['label'] not in ('Other','Mixed') and r['app_replay']['kind']=='UNKNOWN_CONDITION']
    examples=sorted(examples,key=lambda r: r['app_replay']['confidence'],reverse=True)[:8]
    if examples:
        fig,axes=plt.subplots(2,4,figsize=(14,8)); axes=axes.ravel()
        for ax,row in zip(axes,examples):
            image=Image.open(ROOT/row['path']).convert('RGB'); ax.imshow(image)
            for leaf in row['leaves']:
                l,t,r,b=leaf['box']; ax.add_patch(plt.Rectangle((l,t),r-l,b-t,fill=False,edgecolor='#E2A12A',linewidth=1.5))
            ax.set_title(f"Source label: {row['label']}\nOther: {row['app_replay']['confidence']:.1%}\n{row['source']} · {row['id'][-8:]}",fontsize=10); ax.axis('off')
        for ax in axes[len(examples):]: ax.axis('off')
        fig.suptitle('Supported conditions reported as Unknown · highest-confidence examples',x=.04,ha='left'); fig.tight_layout(); fig.savefig(figures/'false_other_examples.png',dpi=140,bbox_inches='tight'); plt.close(fig)
    print(json.dumps(dict(sample_counts=output['sample_counts'],source_summary=summary,bias_paired=paired,tests=output['tests']),indent=2))
    return output


if __name__=='__main__': main()
