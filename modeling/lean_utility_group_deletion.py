"""Fixed, matched partitions of verified historical attention-key indices."""
import hashlib
import numpy as np
import torch


def partition_history(sources, frame, sequence):
    if type(frame) is not int or not isinstance(sequence, str):
        raise ValueError('Invalid read identity')
    eligible=[]
    for source in sources:
        start=source['start']; n=len(source['token_ids']); origin=source['frame']
        if type(start) is not int or type(origin) is not int or start<4096 or origin>=frame:
            raise ValueError('Invalid attributed history')
        if origin < frame-1:
            eligible.extend(range(start,start+n))
    eligible=sorted(eligible)
    if len(eligible)!=len(set(eligible)):
        raise ValueError('Overlapping source spans')
    if len(eligible)!=5120:
        return None
    seed=int.from_bytes(hashlib.sha256(f'lean_utility_group_deletion_v1:{sequence}:{frame}'.encode()).digest()[:16],'big')
    order=np.random.Generator(np.random.PCG64(seed)).permutation(np.asarray(eligible,dtype=np.int64))
    return [sorted(int(i) for i in g) for g in np.split(order,5)]


def delete_group(memory, positions, pointer_count, indices):
    if memory.ndim!=3 or positions.ndim!=3 or memory.shape[:2]!=positions.shape[:2] or memory.shape[1]!=1 or type(pointer_count) is not int or not 0<=pointer_count<=len(memory):
        raise ValueError('Invalid attention geometry')
    spatial=len(memory)-pointer_count
    if len(indices)!=1024 or len(set(indices))!=1024 or any(type(i) is not int or not 4096<=i<spatial for i in indices):
        raise ValueError('Invalid deletion group')
    keep=torch.ones(len(memory),dtype=torch.bool,device=memory.device);keep[torch.tensor(indices,device=memory.device)]=False
    m=memory[keep];p=positions[keep.to(positions.device)]
    if not torch.equal(m[:4096],memory[:4096]) or not torch.equal(p[:4096],positions[:4096]):
        raise ValueError('Conditioning memory changed')
    if pointer_count and (not torch.equal(m[-pointer_count:],memory[-pointer_count:]) or not torch.equal(p[-pointer_count:],positions[-pointer_count:])):
        raise ValueError('Pointer tail changed')
    return m,p
