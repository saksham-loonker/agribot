"""Resumable, identity-bound safetensors ranges; never execute remote model code."""
from pathlib import Path
import hashlib,json,math,shutil,struct
import numpy as np
import requests


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1<<20),b''):h.update(part)
    return h.hexdigest()


class RangeTensors:
    def __init__(self,root,repository,revision):
        assert len(revision)==40 and all(c in '0123456789abcdef' for c in revision)
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.session=requests.Session();self.ranges=[]
        self.url=f'https://huggingface.co/{repository}/resolve/{revision}/model.safetensors'
        identity=dict(repository=repository,revision=revision,url=self.url)
        path=self.root/'identity.json'
        if path.exists():assert json.loads(path.read_text())==identity
        else:path.write_text(json.dumps(identity,indent=2))
        first=self.blob(0,7);n=struct.unpack('<Q',first.read_bytes())[0];assert 0<n<10_000_000
        self.header_path=self.blob(8,7+n);self.header=json.loads(self.header_path.read_bytes());self.data_start=8+n

    def blob(self,start,end):
        assert isinstance(start,int) and isinstance(end,int) and 0<=start<=end
        for a,b,path in self.ranges:
            if start>=a and end<=b:return path
        path=self.root/f'{start}-{end}.bin';proof=path.with_suffix('.json');size=end-start+1
        if path.exists():
            saved=json.loads(proof.read_text());assert path.stat().st_size==size and saved['sha256']==digest(path)
        else:
            partial=path.with_suffix('.partial');present=partial.stat().st_size if partial.exists() else 0
            assert 0<=present<=size and shutil.disk_usage(self.root).free>size-present+(1<<30),'Preserve cache/disk reserve'
            if present<size:
                response=self.session.get(self.url,headers={'Range':f'bytes={start+present}-{end}'},stream=True,timeout=(15,60))
                try:
                    response.raise_for_status();assert response.status_code==206,'Server must honor range; never fetch a whole file accidentally'
                    assert response.headers.get('Content-Range','').startswith(f'bytes {start+present}-{end}/'),'Unexpected response range'
                    with partial.open('ab') as stream:
                        for chunk in response.iter_content(1<<20):
                            if not chunk:continue
                            previous=present;present+=len(chunk);assert present<=size,'Oversized range response';stream.write(chunk)
                            if size>32*(1<<20) and present//(64*(1<<20))!=previous//(64*(1<<20)):print(json.dumps(dict(stage='download_pinned_tensor_range',completed_bytes=present,total_bytes=size)),flush=True)
                    assert present==size,'Incomplete range download; preserve partial bytes for resume'
                finally:response.close()
            partial.replace(path);proof.write_text(json.dumps(dict(start=start,end=end,bytes=size,sha256=digest(path)),indent=2))
        self.ranges.append((start,end,path));return path

    def prefetch(self,names):
        spans=sorted((self.header[name]['data_offsets'][0],self.header[name]['data_offsets'][1]) for name in names)
        merged=[]
        for a,b in spans:
            if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
            else:merged.append([a,b])
        for a,b in merged:self.blob(self.data_start+a,self.data_start+b-1)

    def array(self,name):
        entry=self.header[name];assert entry['dtype']=='F32','Only the explicitly audited FP32 model is supported'
        a,b=entry['data_offsets'];shape=tuple(entry['shape']);assert math.prod(shape)*4==b-a
        start=self.data_start+a;end=self.data_start+b-1
        self.blob(start,end)
        base,_,path=next((a,b,path) for a,b,path in self.ranges if start>=a and end<=b)
        return np.array(np.memmap(path,dtype='<f4',mode='r',offset=start-base,shape=shape),copy=True)

    def rows(self,name,indices):
        entry=self.header[name];assert entry['dtype']=='F32' and len(entry['shape'])==2
        width=entry['shape'][1];row_bytes=width*4;indices=sorted(set(map(int,indices)));assert indices and 0<=indices[0]<=indices[-1]<entry['shape'][0]
        groups=[]
        for i in indices:
            if groups and i-groups[-1][-1]<=8:groups[-1].append(i)
            else:groups.append([i])
        result={}
        for group in groups:
            base=self.data_start+entry['data_offsets'][0]+group[0]*row_bytes;end=self.data_start+entry['data_offsets'][0]+(group[-1]+1)*row_bytes-1
            self.blob(base,end)
            a,_,path=next((a,b,path) for a,b,path in self.ranges if base>=a and end<=b)
            for i in group:result[i]=np.array(np.memmap(path,dtype='<f4',mode='r',offset=base-a+(i-group[0])*row_bytes,shape=(width,)),copy=True)
        return np.stack([result[i] for i in indices])

    def manifest(self):
        result=[]
        for a,b,path in self.ranges:
            sha=digest(path);assert sha==json.loads(path.with_suffix('.json').read_text())['sha256'],'Range changed during model loading'
            result.append(dict(start=a,end=b,bytes=b-a+1,sha256=sha))
        return result
