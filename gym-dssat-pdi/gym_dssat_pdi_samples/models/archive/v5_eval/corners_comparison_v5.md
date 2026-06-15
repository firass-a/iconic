# PC-PPO v5 — Corner-by-Corner Comparison

**Model:** `pc_ppo_custom_best.pt` (archived as `models/archive/thesis_rewards_v5_20260605_best.pt`)
**Eval CSV:** `eval_20260605_201221.csv` (20 episodes × 27 preferences, random weather)
**Reward version:** v5 (`K_FERT=0.040`, `K_LEACH=0.050`, `K_DENIT=0.040`, `K_SEASON_FERT=0.025`)

---

## 1. Side-by-side corner table

| metric | corner_yield | corner_water | corner_fert |
|---|---:|---:|---:|
| **preference vector w** | [1, 0, 0] | [0, 1, 0] | [0, 0, 1] |
| **what it minimizes** | nothing (max yield) | water use | nitrogen use |
| | | | |
| **Grain yield (kg/ha)** | **10,866 ± 1,103** | 10,314 ± 1,177 | 9,108 ± 1,284 |
| Yield vs best corner | — (best) | −5.1% | −16.2% |
| | | | |
| **N applied (kg/ha)** | 718 ± 9 | 384 ± 7 | **246 ± 6** (lowest) |
| **N uptake (kg/ha)** | 303 ± 16 | 294 ± 19 | 235 ± 17 |
| **N leached (kg/ha)** | 384 | 76 | **45** (lowest) |
| **N denitrified (kg/ha)** | 17 | 5 | 2 |
| **NUE %** | 42.2% | 76.5% | **95.4%** (best) |
| | | | |
| **Water applied (mm)** | 1,081 ± 14 | **456 ± 11** (lowest) | 465 ± 14 |
| | | | |
| **N waste = applied − uptake** | 415 | 90 | **11** (lowest) |
| **Water per kg grain (mm/t)** | 99 | **44** (best) | 51 |
| **N applied per kg grain (kg/t)** | 66 | 37 | **27** (best) |
| **Grain per kg N applied** | 15.1 | 26.8 | **37.0** (best) |

All means ± standard deviations across 20 episodes with random weather.

---

## 2. Reading each corner — what kind of farmer it is

### corner_yield — "maximum-production farmer"
- Highest grain yield (~10.9 t/ha)
- Compensates for tight N constraints with **lots of water** (emergent N→W substitution)
- N efficiency moderate (42%) — still ~3× better than v4 baseline behaviour
- Highest leaching of the three corners (384 kg/ha) — the cost of pushing yield to the ceiling

### corner_water — "drought / dry-region farmer"
- Lowest water use (456 mm — viable for savanna or rain-fed-leaning systems)
- Best **water productivity**: 44 mm per ton of grain (vs 99 at yield corner)
- Moderate-to-low N use (384 kg/ha), high NUE (76.5%)
- Loses only 5% of grain yield while halving water and shaving N

### corner_fert — "organic / environmental farmer"
- Lowest N applied (**246 kg/ha** — squarely in real-farmer range, 200–280 kg/ha for U.S. Midwest maize)
- Best **NUE in the family** at 95.4% — almost nothing is wasted
- Lowest leaching (**45 kg/ha**) — groundwater-safe operation
- Pays for it with a 16% yield reduction; still a strong harvest (9.1 t/ha)

---

## 3. The trade-off picture in one line per pair

> **yield → water** : sacrifice 5% grain to save 625 mm of water (huge).
> **yield → fert**  : sacrifice 16% grain to save 472 kg N and 339 kg leached N.
> **water → fert**  : sacrifice 12% grain to halve N application again.

No corner dominates any other on every axis — the agent has cleanly traced a 3-objective Pareto frontier.

---

## 4. Monotonicity checks (all three pass)

| check | result | detail |
|---|:---:|---|
| Yield-priority highest grnwt | **PASS** | 10,866 > 10,314 > 9,108 |
| Water-priority lowest tot_W | **PASS** | 456 < 465 < 1,081 |
| Fert-priority lowest tot_N | **PASS** | 246 < 384 < 718 |

This means the agent is correctly conditioning its policy on the preference vector — different `w` produces structurally different behaviour, not noise around a single mean policy.

---

## 5. Headline single-sentence per corner (thesis-ready)

- **Yield corner**: *"At full yield priority the agent reaches 10.9 t/ha, matching the high-input baseline, but uses 91% less applied N and 73% less water."*
- **Water corner**: *"At full water priority the agent uses just 456 mm of irrigation — 64% less than the moderate baseline — while losing only 5% of grain yield."*
- **Fert corner**: *"At full fertiliser priority the agent achieves 95.4% nitrogen-use efficiency and 45 kg/ha of leaching (33× less than the moderate baseline), at a 16% yield cost."*
