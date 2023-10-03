import numpy as np
import pandas as pd


def hazard_components(s_true, t_true):

    df = (
        pd.DataFrame({'event': s_true, 'time': t_true})
        .groupby('time')
        .agg(['count', 'sum'])
    )

    t = df.index.values
    d = df[('event', 'sum')].values
    c = df[('event', 'count')].values
    n = np.sum(c) - np.cumsum(c) + c

    return t, d, n


def kaplan_meier(s_true, t_true, t_new=None):

    t, d, n = hazard_components(s_true, t_true)

    m = np.cumprod(1 - np.divide(
        d, n,
        out=np.zeros(len(d)),
        where=n > 0
    ))
    
    v = (m ** 2) * np.cumsum(np.divide(
        d, n * (n - d),
        out=np.zeros(len(d)),
        where=n * (n - d) > 0
    ))

    if t_new is not None:
        return interpolate(t, m, t_new)
    else:
        return t, m, v


def nelson_aalen(s_true, t_true, t_new=None):

    t, d, n  = hazard_components(s_true, t_true)

    m = np.cumsum(np.divide(
        d, n,
        out=np.zeros(len(d)),
        where=n > 0
    ))
    
    v = np.cumsum(np.divide(
        (n - d) * d,
        (n - 1) * (n ** 2),
        out=np.zeros(len(d)),
        where=(n - 1) > 0
    ))
        
    if t_new is not None:
        return interpolate(t, m, t_new)
    else:
        return t, m, v


def interpolate(x, y, new_x, method='pad'):
    
    # s = pd.Series(data=y, index=x)
    # new_y = (s
    #     .reindex(s.index.union(new_x).unique())
    #     .interpolate(method=method)[new_x]
    #     .values
    # )
    
    # return new_y

    return np.interp(new_x, x, y)


def bootstrappable(func):

    def bootstrap_func(*args, n_bootstrap_samples=None, random_state=None, ci=95, **kwargs):

        estimate = func(*args, **kwargs)

        if n_bootstrap_samples is not None:

            assert type(n_bootstrap_samples) is int, n_bootstrap_samples

            rs = np.random.RandomState(seed=random_state)
            
            N = len(args[0])
            indices = np.arange(N)

            samples = []

            for _ in range(n_bootstrap_samples):

                sample_indices = rs.choice(indices, len(indices), replace=True)

                func_args = [
                    arg[sample_indices]
                    if (hasattr(arg, '__len__') and (len(arg) == N))
                    else arg
                    for arg in args
                ]
                
                func_kwargs = {
                    kw: arg[sample_indices]
                    if (hasattr(arg, '__len__') and (len(arg) == N))
                    else arg
                    for kw, arg in kwargs.items()
                }

                samples.append(func(*func_args, **func_kwargs))

            return (
                estimate,
                np.percentile(samples, 50 - ci / 2., axis=0),
                np.percentile(samples, 50 + ci / 2., axis=0)
            )

        else:

            return estimate

    return bootstrap_func


@bootstrappable
def xAUCt(s_test, t_test, pred_risk, times, pos_group=None, neg_group=None):

    # NOTE: enter groups pos_group and neg_group for xAUC_t; omit for AUC_t

    # pred_risk can be 1d (static) or 2d (time-varying)
    if len(pred_risk.shape) == 1:
        pred_risk = pred_risk[:, np.newaxis]

    # positives: s_test = 1 & t_test =< t
    pos = (t_test[:, np.newaxis] <= times[np.newaxis, :]) & s_test[:, np.newaxis]

    if pos_group is not None:
        pos = pos & pos_group[:, np.newaxis]
    
    # negatives: t_test > t
    neg = (t_test[:, np.newaxis] > times[np.newaxis, :])

    if neg_group is not None:
        neg = neg & neg_group[:, np.newaxis]

    valid = pos[:, np.newaxis, :] & neg[np.newaxis, :, :]
    correctly_ranked = valid & (pred_risk[:, np.newaxis, :] > pred_risk[np.newaxis, :, :])

    return np.sum(correctly_ranked, axis=(0, 1)) / np.sum(valid, axis=(0, 1))


@bootstrappable
def xAPt(s_test, t_test, pred_risk, times, pos_group=None, neg_group=None, return_prevalence=False):

    ap = []
    prev = []

    for idx, time in enumerate(times):

        # pred_risk can be 1d (static) or 2d (time-varying)
        if len(pred_risk.shape) == 1:
            prt = pred_risk
        else:
            prt = pred_risk[:, idx]

        recall, precision, threshold, prevalence = xPRt(
            s_test, t_test, prt, time,
            pos_group=pos_group, neg_group=neg_group
        )
        
        ap.append(-1 * np.sum(np.diff(recall) * np.array(precision)[:-1]))
        prev.append(prevalence)

    return (np.array(ap), np.array(prev)) if return_prevalence else np.array(ap)


def xROCt(s_test, t_test, pred_risk, time, pos_group=None, neg_group=None):

    # NOTE: enter groups pos_group and neg_group for xROC_t; omit for ROC_t

    threshold = np.append(np.sort(pred_risk), np.infty)

    # positives: s_test = 1 & t_test =< t
    pos = (t_test < time) & s_test

    if pos_group is not None:
        pos = pos & pos_group
    
    # negatives: t_test > t
    neg = (t_test > time)

    if neg_group is not None:
        neg = neg & neg_group

    # prediction
    pred = pred_risk[:, np.newaxis] > threshold[np.newaxis, :]

    tpr = np.sum(pred & pos[:, np.newaxis], axis=0) / np.sum(pos)
    fpr = np.sum(pred & neg[:, np.newaxis], axis=0) / np.sum(neg)

    return tpr, fpr, threshold


def xPRt(s_test, t_test, pred_risk, time, pos_group=None, neg_group=None):

    threshold = np.append(np.sort(pred_risk), np.infty)

    # positives: s_test = 1 & t_test =< t
    pos = (t_test < time) & s_test

    if pos_group is not None:
        pos = pos & pos_group
    
    # negatives: t_test > t
    neg = (t_test > time)

    if neg_group is not None:
        neg = neg & neg_group

    # prediction
    pred = pred_risk[:, np.newaxis] > threshold[np.newaxis, :]

    tps = np.sum(pred & pos[:, np.newaxis], axis=0)
    fps = np.sum(pred & neg[:, np.newaxis], axis=0)

    positives = np.sum(pos)
    negatives = np.sum(neg)

    recall = tps / positives
    precision = np.divide(tps, tps + fps, out=np.ones_like(tps, dtype=float), where=(tps + fps) > 0)

    prevalence = positives / (positives + negatives)

    return recall, precision, threshold, prevalence


def ipc_weights(s_train, t_train, s_test, t_test, tau=None):

    if tau == 'auto':
        mask = t_test < t_train[s_train == 1].max()
        #mask = t_test < t_train[s_train == 0].max()
    
    elif tau is not None:
        mask = t_test < tau

    else:
        mask = np.ones_like(t_test, dtype=bool)

    pc = kaplan_meier(1 - s_train, t_train, t_test)
    pc[s_test == 0] = 1.

    w = 1. / pc
    w[~mask] = 0.

    return w


@bootstrappable
def xCI(s_test, t_test, pred_risk,
        weights=None,
        pos_group=None, neg_group=None,
        return_num_valid=False,
        tied_tol=1e-8):

    w = weights if weights is not None else np.ones_like(s_test)

    mask1 = (s_test == 1)

    if pos_group is not None:
        mask1 = mask1 & pos_group

    w = w[mask1, np.newaxis]

    mask2 = np.ones_like(s_test, dtype=bool)

    if neg_group is not None:
        mask2 = mask2 & neg_group

    valid = t_test[mask1, np.newaxis] < t_test[np.newaxis, mask2]

    risk_diff = pred_risk[mask1, np.newaxis] - pred_risk[np.newaxis, mask2]

    correctly_ranked = valid & (risk_diff > tied_tol)
    tied = valid & (np.abs(risk_diff) <= tied_tol)

    num_valid = np.sum((w ** 2) * valid)
    ci = np.sum((w ** 2) * (correctly_ranked + 0.5 * tied)) / num_valid

    return (ci, num_valid) if return_num_valid else ci


@bootstrappable
def xxCI(s_test, t_test, pred_risk, weights=None, pos_group=None, neg_group=None):

    m1, n1 = xCI(
        s_test, t_test, pred_risk,
        weights=weights,
        pos_group=pos_group, neg_group=neg_group,
        return_num_valid=True
    )
    
    m2, n2 = xCI(
        s_test, t_test, pred_risk,
        weights=weights,
        pos_group=neg_group, neg_group=pos_group,
        return_num_valid=True
    )
    
    return (m1 * n1 + m2 * n2) / (n1 + n2)
