# Smart Farm DSS — CAPQL Interface

Farmer-oriented web UI for **pretrained CAPQL v2** on DSSAT maize.  
**No mock data** — simulations and Pareto points come from real CAPQL + DSSAT runs.

## Architecture

```text
interface/web     React + TypeScript + Tailwind + Recharts  (port 5173)
       │  /api proxy
interface/api     FastAPI  (port 8000)
       │
capql_inference.py + CAPQLEnv + actor.pt + DSSAT  (Docker)
```

## Prerequisites

- Docker Desktop (for API + DSSAT)
- Node.js 20+ (for frontend)
- Trained weights: `models/capql_v2/actor.pt`

## Run (two terminals)

### 1. API (Docker)

```powershell
cd iconic
.\interface\run_api_docker.ps1
```

API docs: http://localhost:8000/docs

### 2. Frontend (local)

```powershell
cd iconic\interface\web
npm install
npm run dev
```

Open: http://localhost:5173

## Usage

1. Open **Simulation** → set preference sliders (Yield / N efficiency / Water saving)
2. Click **Run simulation** (~1–3 min, real DSSAT season)
3. View **Dashboard**, **Recommendations**, **Reports**, **Trade-offs**

Pareto scatter uses `capql_v2_eval_results.csv` (real eval data).

**Weather / rain:** Simulations use WGEN with `weather/UFGA.CLI` (Gainesville climate). Restart the API after pulling env changes. Check `GET /health` → `weather_cli_loaded: true`. Re-run a simulation to see rain in Reports and Recommendations.

## Files

| Path | Role |
|------|------|
| `gym-dssat-pdi/gym_dssat_pdi_samples/capql_inference.py` | Run one season, day-by-day log |
| `interface/api/main.py` | FastAPI jobs + pareto |
| `gym-dssat-pdi/gym_dssat_pdi_samples/weather_config.py` | UFGA.CLI path for WGEN rain |
| `interface/web/src/services/api.ts` | Frontend API client |

## Notes

- Crop, soil, and planting are fixed by DSSAT config (read-only in UI).
- Recommendation “reason” text is rule-based interpretation, not RL output.
- Only one simulation runs at a time (DSSAT worker pool size = 1).
