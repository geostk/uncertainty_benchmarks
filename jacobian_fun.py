def jacobian(x, func):
    """
    estimate Jacobian
    """
    nx = len(x)  # degrees of freedom
    x = np.array(x)
    LOGGER.debug(x)
    nnx = x[0].size
    assert all(nnx == _.size for _ in x)
    j = None  # matrix of zeros
    delta = np.eye(nx) * DELTA
    LOGGER.debug(delta)
    for d in delta:
        df = np.array(func(*(x.T + d).T)) - np.array(func(*(x.T - d).T))
        df = np.reshape(df / DELTA / 2, (-1, 1))
        if j is None:
            j = df
            LOGGER.debug(j)
        else:
            j = np.append(j, df, axis=1)  # derivatives df/d_n
    return j
