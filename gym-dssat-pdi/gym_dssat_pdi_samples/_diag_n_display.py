from collections import Counter

from capql_inference import load_actor, run_season

actor = load_actor()
r = run_season([1 / 3, 1 / 3, 1 / 3], seed=123, actor=actor, weather_id='wgen-nasa-algiers')
buckets = Counter()
for d in r.days:
    n = d.nitrogen_kg_ha
    if n < 0.1:
        buckets['0'] += 1
    elif n < 1:
        buckets['0.1-0.9'] += 1
    elif n < 5:
        buckets['1-4.9'] += 1
    else:
        buckets['5+'] += 1
display_zero = sum(1 for d in r.days if round(d.nitrogen_kg_ha) == 0)
print('total_N', r.total_n_kg_ha)
print('buckets', dict(buckets))
print('days showing 0 after toFixed(0):', display_zero, '/', len(r.days))
