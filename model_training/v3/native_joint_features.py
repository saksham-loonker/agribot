"""Join native-resolution encoder rows and old texture rows by linked SHA IDs."""
import numpy as np


def join_native_cnn(native,keys,links,cnn):
    native=np.asarray(native)
    if native.shape!=(len(keys),1024) or native.dtype!=np.float32 or len(set(keys))!=len(keys) or not np.isfinite(native).all():raise ValueError('Invalid native feature identity table')
    position={key:i for i,key in enumerate(keys)};result=[]
    for link in links:
        key=link['input_sha256'];parent=link['parent_224_input_sha256']
        if key not in position or parent not in cnn:raise ValueError('Missing linked native/parent feature identity')
        texture=np.asarray(cnn[parent])
        if texture.shape!=(768,) or texture.dtype!=np.float32 or not np.isfinite(texture).all():raise ValueError('Invalid linked CNN representation')
        result.append(np.concatenate((native[position[key]],texture)))
    if not result:raise ValueError('No linked native/CNN inputs')
    return np.stack(result)
