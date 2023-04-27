
def algorithm_nearest(curr_ts, prev_ts, prev_msg, next_ts, next_msg):
    if curr_ts - prev_ts <= next_ts - curr_ts:
        return prev_msg
    else:
        return next_msg


def always_return_something(f):
    def wrapper(curr_ts, prev_ts, prev_msg, next_ts, next_msg):
        if prev_ts is None:
            return next_msg
        if next_ts is None:
            return prev_msg
        return f(curr_ts, prev_ts, prev_msg, next_ts, next_msg)
    return wrapper


ALGORITHMS = {
    'previous': lambda ct, pt, pm, nt, nm: pm,
    'next': lambda ct, pt, pm, nt, nm: nm,
    'nearest': always_return_something(algorithm_nearest),
    'exact': lambda ct, pt, pm, nt, nm: None,
}


def align(target_iterator, *other_iterators, **options):
    # prepare parameters
    default_alg = options.get('algorithm', 'nearest')
    other_iterators = [
        it if isinstance(it, tuple) else (it, default_alg)
            for it in other_iterators
    ]

    # prepare last_infos
    last_infos = []
    for it, alg in other_iterators:
        msg = next(it, None)
        ts = msg[0] if msg is not None else None
        last_infos.append((ts, msg, None, None))

    # main iterator
    for target_msg in target_iterator:
        target_ts = target_msg[0]
        res = [target_msg]

        for idx, (it, alg) in enumerate(other_iterators):
            # make sure last_two_ts <= target_ts <= last_one_ts unless one is None
            last_one_ts, last_one_msg, last_two_ts, last_two_msg = last_infos[idx]

            while last_one_ts and last_one_ts < target_ts:
                last_two_ts, last_two_msg = last_one_ts, last_one_msg
                last_one_msg = next(it, None)
                last_one_ts = last_one_msg[0] if last_one_msg is not None else None
            last_infos[idx] = (last_one_ts, last_one_msg,
                               last_two_ts, last_two_msg)

            # exactly the same! this is the eager case for 'previous' algorithm
            if last_one_ts == target_ts:
                res.append(last_one_msg)

            # use the algorithm
            else:
                assert (last_two_ts or 0) < target_ts < (last_one_ts or 1e99)
                res.append(ALGORITHMS[alg](target_ts, last_two_ts,
                                           last_two_msg, last_one_ts,
                                           last_one_msg))
        yield res
