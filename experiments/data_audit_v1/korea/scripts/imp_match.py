"""Import series vs Danawa import lines (KAIDA registration data): assign Danawa lines to our series,
infer the consistent definition (set of lines) from the data, and explain monthly mismatches."""
import re, itertools
import pandas as pd, numpy as np

STOP = {'the', 'new', 'all', 'all-new'}
ALIAS = {('Mini', 'hatch'): 'Cooper', ('Lincoln', 'mkx'): 'Nautilus', ('Audi', 'etron'): 'Q8 e-tron', ('Toyota', '86'): 'GR86',
         ('Rolls-Royce', 'cullinan'): 'Cullinan Series2', ('Rolls-Royce', 'ghost'): 'Ghost Series II', ('Rolls-Royce', 'phantom'): 'Phantom Series II', ('Volvo', 'ex40'): 'XC40'}
NAME_ALIAS = {'polestar2': ('Polestar', '2'), 'polestar4': ('Polestar', '4'), 'aviator': ('Lincoln', 'Aviator')}

def toks(s):
    s = s.lower().replace('-', ' ')
    return [t for t in s.split() if t not in STOP]

def subseq(a, b):
    n = len(a)
    return any(b[i:i + n] == a for i in range(len(b) - n + 1))

def build(k, dn):
    """k: our Korea rows; dn: Danawa model rows (all)."""
    di = dn[dn.nat == 'export'].copy()
    imp_brands = sorted(set(k.Brand) - {'Hyundai', 'Genesis', 'Kia', 'KG Mobility', 'Renault Korea'})
    ours_models = {b: sorted(k[k.Brand == b].Model.unique()) for b in imp_brands}
    # brand id inference: exact token-equality matches
    votes = {}
    for r in di.drop_duplicates(['brand_id', 'name']).itertuples():
        tn = toks(r.name) if isinstance(r.name, str) else []
        for b, ms in ours_models.items():
            for m in ms:
                if toks(m) == tn or ''.join(toks(m)) == ''.join(tn):
                    votes.setdefault(r.brand_id, {}).setdefault(b, 0)
                    votes[r.brand_id][b] += 1
    bid2brand = {bid: max(v, key=v.get) for bid, v in votes.items()}
    # Chevrolet imports appear on the import page too; our Chevrolet series are audited vs domestic (GM Korea) figures
    assign = {}
    for r in di.drop_duplicates(['brand_id', 'name']).itertuples():
        if not isinstance(r.name, str): continue
        tn = toks(r.name)
        if ''.join(tn) in NAME_ALIAS:
            assign[(r.brand_id, r.name)] = NAME_ALIAS[''.join(tn)]; continue
        b = bid2brand.get(r.brand_id)
        if b is None: continue
        al = ALIAS.get((b, ''.join(tn)))
        if al: assign[(r.brand_id, r.name)] = (b, al); continue
        best = None
        for m in ours_models[b]:
            tm = toks(m)
            if tm == tn or ''.join(tm) == ''.join(tn):
                sc = (100, 1)
            elif subseq(tm, tn):
                sc = (len(tm), 1 if tn[:len(tm)] == tm else 0)
            else:
                continue
            if best is None or sc > best[0]: best = (sc, m)
        if best: assign[(r.brand_id, r.name)] = (b, best[1])
    di['our'] = [assign.get((bi, nm)) for bi, nm in zip(di.brand_id, di.name)]
    di = di.dropna(subset=['our'])
    lines = di.groupby(['our', 'period', 'name']).sales.sum()
    return di, lines, bid2brand

def infer(k, di):
    """Return dict (Brand,Model)->dict(lines, defn, per-month official & explanation)."""
    ours = k.set_index(['Brand', 'Model', 'Month']).Sales
    res = {}
    for key, g in di.groupby('our'):
        b, m = key
        if b == 'Chevrolet': continue
        piv = g.pivot_table(index='period', columns='name', values='sales', aggfunc='sum').fillna(0.0)
        names = list(piv.columns)
        expl = {}
        incl = {n: [0, 0] for n in names}
        for p in piv.index:
            o = ours.get((b, m, p), np.nan)
            if pd.isna(o): continue
            vals = piv.loc[p]
            act = [n for n in names if vals[n] > 0]
            sol = None
            if len(act) <= 7:
                for tot in range(0, 2 * len(act) + 1):
                    for mults in itertools.product(range(3), repeat=len(act)):
                        if sum(mults) != tot: continue
                        if abs(sum(mu * vals[n] for mu, n in zip(mults, act)) - o) < 0.5:
                            sol = {n: mu for mu, n in zip(mults, act)}; break
                    if sol is not None: break
            expl[p] = sol
            if sol is not None:
                for n in act:
                    incl[n][1] += 1
                    if sol.get(n, 0) >= 1: incl[n][0] += 1
        defn = [n for n in names if incl[n][1] > 0 and incl[n][0] / incl[n][1] >= 0.5]
        if not defn:  # fall back to exact-name lines
            defn = [n for n in names if toks(n) == toks(m)] or names
        off = piv[defn].sum(axis=1)
        res[key] = dict(lines=names, defn=defn, official=off, expl=expl, piv=piv, incl=incl)
    return res
