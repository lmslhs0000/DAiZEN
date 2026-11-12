import math
import re
from collections import defaultdict

import numpy as np
import pandas as pd


KEYS = ['Country', 'Brand', 'Model']
FEATURE_COLUMNS = ['country_peer_growth_3v9', 'brand_peer_growth_3v9',
                   'brand_share_change_3v9_pp']
MEMBERSHIP_COLUMNS = KEYS + ['origin', 'first_month', 'window_present_months',
                            'window_observed_months', 'cohort_status', 'prior9_sum', 'recent3_sum']
OUTPUT_COLUMNS = KEYS + ['origin', 'own_history_status', 'prior9_sum', 'recent3_sum',
    'country_candidate_series', 'country_stable_series', 'brand_candidate_series',
    'brand_stable_series', 'country_peer_count', 'brand_peer_count',
    'country_peer_prior9_sum', 'country_peer_recent3_sum', 'brand_peer_prior9_sum',
    'brand_peer_recent3_sum', 'brand_prior9_sum', 'brand_recent3_sum'] + FEATURE_COLUMNS + [
    'country_context_status', 'brand_context_status', 'share_context_status', 'context_status']


def _month(value):
    text = str(value)
    if not re.fullmatch(r'\d{4}-\d{2}', text):
        raise ValueError('Month must be YYYY-MM')
    return pd.Period(text, freq='M')


def _validated_copy(data):
    required = KEYS + ['Month', 'Sales']
    if not set(required).issubset(data.columns):
        raise ValueError('Required columns: Country, Month, Brand, Model, Sales')
    frame = data[required].copy()
    for name in KEYS:
        if not frame[name].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f'Invalid series key: {name}')
    text = frame.Month.astype(str)
    if not text.str.fullmatch(r'\d{4}-\d{2}').all():
        raise ValueError('Month must be YYYY-MM')
    frame['Month'] = pd.PeriodIndex(text, freq='M')
    values = pd.to_numeric(frame.Sales, errors='coerce').astype(float)
    nonblank = frame.Sales.notna() & frame.Sales.astype(str).str.strip().ne('')
    if (nonblank & values.isna()).any() or np.isinf(values).any() or values.lt(0).any():
        raise ValueError('Sales must be finite and nonnegative, or missing')
    frame['Sales'] = values
    if frame.duplicated(KEYS + ['Month']).any():
        raise ValueError('Duplicate series/month')
    return frame


def _membership(history, origin):
    if history.empty:
        return pd.DataFrame(columns=MEMBERSHIP_COLUMNS)
    months = pd.period_range(origin - 11, origin, freq='M')
    first = history.groupby(KEYS, sort=True, observed=True).Month.min()
    window = history[history.Month.ge(months[0])]
    values = window.pivot(index=KEYS, columns='Month', values='Sales').reindex(
        index=first.index, columns=months)
    present = window.groupby(KEYS, observed=True).size().reindex(first.index, fill_value=0)
    rows = []
    for key, sales in values.iterrows():
        observed = int(sales.notna().sum())
        state = ('OK' if observed == 12 else 'INSUFFICIENT_HISTORY'
                 if first.loc[key] > months[0] else 'INCOMPLETE_HISTORY')
        rows.append({**dict(zip(KEYS, key)), 'origin': str(origin), 'first_month': str(first.loc[key]),
                     'window_present_months': int(present.loc[key]), 'window_observed_months': observed,
                     'cohort_status': state,
                     'prior9_sum': math.fsum(sales.iloc[:9]) if state == 'OK' else np.nan,
                     'recent3_sum': math.fsum(sales.iloc[9:]) if state == 'OK' else np.nan})
    return pd.DataFrame(rows, columns=MEMBERSHIP_COLUMNS)


def _sum(rows, name):
    value = math.fsum(row[name] for row in rows)
    if not math.isfinite(value):
        raise ValueError('Nonfinite group total')
    return value


def _growth(peers, level):
    prior = _sum(peers, 'prior9_sum')
    recent = _sum(peers, 'recent3_sum')
    if not peers:
        return np.nan, f'NO_{level}_PEERS', prior, recent
    if prior == 0:
        return np.nan, f'ZERO_{level}_REFERENCE', prior, recent
    value = (recent / 3) / (prior / 9) - 1
    if not math.isfinite(value):
        raise ValueError('Nonfinite peer growth')
    return value, 'OK', prior, recent


def build_context_snapshot(data, origin):
    origin = _month(origin)
    frame = _validated_copy(data)
    history = frame[frame.Month.le(origin)]
    membership = _membership(history, origin)
    country_groups, brand_groups = defaultdict(list), defaultdict(list)
    for row in membership[membership.cohort_status.eq('OK')].to_dict('records'):
        country_groups[row['Country']].append(row)
        brand_groups[(row['Country'], row['Brand'])].append(row)
    country_counts = membership.groupby('Country', observed=True).size()
    brand_counts = membership.groupby(['Country', 'Brand'], observed=True).size()
    output = []
    for member in membership.to_dict('records'):
        country, brand, model = (member[name] for name in KEYS)
        country_panel = country_groups[country]
        brand_panel = brand_groups[(country, brand)]
        country_peers = [r for r in country_panel if (r['Brand'], r['Model']) != (brand, model)]
        brand_peers = [r for r in brand_panel if r['Model'] != model]
        country_growth, country_state, country_prior, country_recent = _growth(country_peers, 'COUNTRY')
        brand_growth, brand_state, brand_prior, brand_recent = _growth(brand_peers, 'BRAND')
        panel_prior = _sum(brand_panel, 'prior9_sum')
        panel_recent = _sum(brand_panel, 'recent3_sum')
        share = np.nan
        if not brand_peers:
            share_state = 'NO_BRAND_PEERS'
        elif panel_prior == 0 or panel_recent == 0:
            share_state = 'ZERO_BRAND_SHARE_DENOMINATOR'
        else:
            share_state = 'OK'
            share = 100 * (member['recent3_sum'] / panel_recent - member['prior9_sum'] / panel_prior)
        if member['cohort_status'] != 'OK':
            country_growth = brand_growth = share = np.nan
            country_state = brand_state = share_state = 'OWN_HISTORY_UNAVAILABLE'
        available = sum(math.isfinite(value) for value in (country_growth, brand_growth, share))
        output.append({**{name: member[name] for name in KEYS}, 'origin': str(origin),
            'own_history_status': member['cohort_status'], 'prior9_sum': member['prior9_sum'],
            'recent3_sum': member['recent3_sum'], 'country_candidate_series': int(country_counts[country]),
            'country_stable_series': len(country_panel), 'brand_candidate_series': int(brand_counts[(country, brand)]),
            'brand_stable_series': len(brand_panel), 'country_peer_count': len(country_peers),
            'brand_peer_count': len(brand_peers), 'country_peer_prior9_sum': country_prior,
            'country_peer_recent3_sum': country_recent, 'brand_peer_prior9_sum': brand_prior,
            'brand_peer_recent3_sum': brand_recent, 'brand_prior9_sum': panel_prior,
            'brand_recent3_sum': panel_recent, 'country_peer_growth_3v9': country_growth,
            'brand_peer_growth_3v9': brand_growth, 'brand_share_change_3v9_pp': share,
            'country_context_status': country_state, 'brand_context_status': brand_state,
            'share_context_status': share_state,
            'context_status': 'OK' if available == 3 else 'PARTIAL' if available else 'UNAVAILABLE'})
    features = pd.DataFrame(output, columns=OUTPUT_COLUMNS)
    return features, membership
