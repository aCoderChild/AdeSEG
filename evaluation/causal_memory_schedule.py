"""Predeclared coverage for offline causal reads; never a memory-frame policy."""


def fixed_causal_schedule(presence, prompt):
    if not presence or not 0 <= prompt < len(presence)-1 or not presence[prompt]:
        raise ValueError('A present prompt and at least one later frame are required')
    if any(type(value) is not bool for value in presence):
        raise ValueError('Presence must be a boolean manifest-frame sequence')
    selected = {}

    def add(frame, stratum):
        if not prompt < frame < len(presence):
            raise ValueError('Selected target is outside the post-prompt video')
        selected.setdefault(frame, []).append(stratum)

    start, stop = prompt + 1, len(presence) - 1
    for numerator in (1, 3, 5, 7, 9):
        # Integer arithmetic fixes rounding and avoids sample-dependent tuning.
        add(start + (stop-start)*numerator//10, f'uniform_{numerator}/10')
    absent_runs, returns = [], []
    cursor = start
    while cursor < len(presence):
        if presence[cursor]:
            cursor += 1
            continue
        low = cursor
        while cursor < len(presence) and not presence[cursor]:
            cursor += 1
        high = cursor  # exclusive
        midpoint = (low + high - 1)//2
        add(midpoint, 'gt_empty_midpoint')
        absent_runs.append({'start': low, 'stop': high, 'midpoint': midpoint})
        if high < len(presence):
            end = high
            while end < len(presence) and presence[end]:
                end += 1
            followup = list(range(high, min(high+3, end)))
            for offset, frame in enumerate(followup):
                add(frame, f'reappearance_t{offset}')
            returns.append({'return_frame': high, 'absence_start': low,
                            'absence_length': high-low, 'episode_stop': end,
                            'evaluated_followup': followup})
    # Cover short positive episodes that the uniform schedule could miss.
    present_runs = []
    cursor = start
    while cursor < len(presence):
        if not presence[cursor]:
            cursor += 1
            continue
        low = cursor
        while cursor < len(presence) and presence[cursor]:
            cursor += 1
        midpoint = (low + cursor - 1)//2
        add(midpoint, 'gt_present_midpoint')
        present_runs.append({'start': low, 'stop': cursor, 'midpoint': midpoint})
    return {'targets': [{'frame': frame, 'strata': sorted(set(strata)),
                          'present': presence[frame]} for frame, strata in sorted(selected.items())],
            'absent_runs': absent_runs, 'present_runs': present_runs, 'return_events': returns,
            'num_frames': len(presence), 'prompt': prompt}
