"""Compare static Python replay against actual compiled app Kotlin classes."""
import json, os, subprocess
from pathlib import Path
from evaluate_public import ROOT, OUT, LABELS, MANIFEST, replay

snapshot=(OUT/'predictions.jsonl').read_text()
lines=snapshot.splitlines()
if not snapshot.endswith('\n'): lines=lines[:-1]
rows=[json.loads(line) for line in lines]
work=OUT/'kotlin-verification'; work.mkdir(exist_ok=True)
jdk=next((ROOT/'.toolchains/java').glob('jdk-*'))
stdlib=next((ROOT/'.toolchains/gradle/caches/modules-2/files-2.1/org.jetbrains.kotlin/kotlin-stdlib/1.9.25').rglob('*.jar'))
classpath=os.pathsep.join(map(str,[work,stdlib,ROOT/'agribot_android_app/android/domain/build/classes/kotlin/main',ROOT/'agribot_android_app/android/ml/build/tmp/kotlin-classes/debug']))
subprocess.run([str(jdk/'bin/javac.exe'),'-cp',classpath,'-d',str(work),str(Path(__file__).with_name('VerifyKotlinReplay.java'))],check=True)
fixture=work/'input.tsv'
with fixture.open('w') as stream:
    for row in rows:
        if not row['leaves']: stream.write(row['id']+'\t0\n')
        for leaf in row['leaves']:
            members='|'.join(','.join(map(str,z)) for z in leaf['member_logits'])
            stream.write(f"{row['id']}\t{row['quality_weight']}\t{leaf['score']}\t{members}\n")
results=[]
for bias in [.75,0.]:
    output=work/f'bias-{bias}.tsv'
    subprocess.run([str(jdk/'bin/java.exe'),'-cp',classpath,'VerifyKotlinReplay',str(fixture),str(output),str(bias),str(MANIFEST['classifier']['temperature'])],check=True)
    jvm={}
    for line in output.read_text().splitlines():
        identity,kind,index,frames,confidence,partial=line.split('\t')
        jvm[identity]=dict(kind=kind,label=None if index=='null' else LABELS[int(index)],frames=int(frames),confidence=float(confidence),partial=partial=='true')
    mismatches=[]; max_diff=0.
    for row in rows:
        expected=row['app_replay'] if bias==.75 else replay(row['leaves'],row['quality_weight'],bias=bias)
        actual=jvm[row['id']]
        for field in ('kind','label','frames','partial'):
            if expected[field]!=actual[field]: mismatches.append(dict(id=row['id'],field=field,python=expected[field],kotlin=actual[field]))
        if expected['confidence'] is not None:
            difference=abs(expected['confidence']-actual['confidence']); max_diff=max(max_diff,difference)
            if difference>1e-5: mismatches.append(dict(id=row['id'],field='confidence',difference=difference))
    results.append(dict(bias=bias,images=len(rows),mismatches=mismatches,max_confidence_difference=max_diff))
(OUT/'kotlin_replay_verification.json').write_text(json.dumps(results,indent=2))
assert all(not result['mismatches'] for result in results), 'Kotlin/Python replay disagreement'
print(json.dumps(results,indent=2))
