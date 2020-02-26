def ev(data, biascorr=True):
    """Computes the amount of variance in a voxel's response that can be explained by the
    mean response of that voxel over multiple repetitions of the same stimulus.

    If [biascorr], the explainable variance is corrected for bias, and will have mean zero
    for random datasets.

    Data is assumed to be a 2D matrix: time x repeats.
    """
    ev = 1 - (data.T - data.mean(1)).var() / data.var()
    if biascorr:
        return ev - ((1 - ev) / (data.shape[1] - 1.0))
    else:
        return ev
