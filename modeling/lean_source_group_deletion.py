"""Partition eligible raw Lean memory keys into complete older-source blocks."""
def partition_history(sources, frame, sequence):
    if type(frame) is not int or not isinstance(sequence,str):
        raise ValueError('Invalid read identity')
    older=[]
    for source in sources:
        start=source['start']; origin=source['frame']; n=len(source['token_ids'])
        if type(start) is not int or type(origin) is not int or start<4096 or origin>=frame:
            raise ValueError('Invalid attributed history')
        if origin<frame-1: older.append((origin,start,n))
    older.sort()
    if len(older)!=5 or any(n!=1024 for _,_,n in older):return None
    if len({x[0] for x in older})!=5:raise ValueError('Duplicate source frame')
    groups=[list(range(start,start+n)) for _,start,n in older]
    if len({x for g in groups for x in g})!=5120:raise ValueError('Overlapping source spans')
    return groups
