# Collaborator setup — commands

This walks through what to run **after cloning** the GitHub repo (for example [`firass-a/iconic`](https://github.com/firass-a/iconic.git)). Layout matches upstream **gym-dssat** development: Docker is the usual way to get a working DSSAT stack on Linux.

---

## Prerequisites

- **Docker** installed and runnable (`docker --version`).
- **Git**.
- Enough disk space for the image build and DSSAT runtime files.

Optional: SSH key or a **GitHub Personal Access Token** if you fork or use HTTPS with 2FA.

---

## 1. Clone

```bash
git clone https://github.com/firass-a/iconic.git
cd iconic
```

If this repository tracks **Git submodules** (DSSAT sources/data) and you need them for native builds—not required for the prebuilt Docker workflow—initialize once:

```bash
git submodule update --init --recursive
```

---

## 2. Build the gym-dssat Docker image (Debian Bookworm)

Run this from the **repository root** (the directory that contains `docker_recipes/`).

```bash
docker build -f docker_recipes/Dockerfile_Debian_Bookworm -t gym-dssat:debian-bookworm docker_recipes
```

Tag `gym-dssat:debian-bookworm` is what the sample commands below assume.

Smoke test:

```bash
docker run --rm gym-dssat:debian-bookworm
```

---

## 3. Run sample scripts inside the container

Mount the checkout so edits on the host are visible in the container. Replace the host path if your clone lives elsewhere.

```bash
HOST_REPO="$(pwd)"

docker run --rm -it -u root \
  -v "${HOST_REPO}:/workspace" \
  -w /workspace/gym-dssat-pdi/gym_dssat_pdi_samples \
  gym-dssat:debian-bookworm bash
```

You are now in **`gym_dssat_pdi_samples`** (`-w`). Stay in this directory for imports (`pc_env` loads sibling modules).

### Stable-Baselines3 + PyTorch (install once per fresh container)

The base image pins **NumPy 1.24.x** for `gym-dssat-pdi`. Install SB3 stack in a compatible way (**as root** avoids user-site clashes; pinning avoids NumPy/Matplotlib breaks):

```bash
python3 -m pip install --no-cache-dir numpy==1.24.1 "pandas>=2,<3" "gymnasium>=0.29.1,<1.3.0"
python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --no-cache-dir stable-baselines3 -c <(echo "numpy==1.24.1")
```

---

## 4. Sample scripts (order)

| Script | Purpose |
|--------|---------|
| `01_rule_based_baselines.py` | Rule-based baselines / SmartFarm wrapper diagnostics |
| `02_smart_farm_env.py` | SoS wrapper (imported by others; conflicts here break everything) |
| `pc_env.py` | Preference-conditioned Gymnasium adapter; `python3 pc_env.py` runs a smoke test |
| `03_sb3_sanity_check.py` | Short PPO smoke test (~512 env steps) |
| `04_pc_ppo_quick_train.py` | ~50k-step PPO + preference-differentiation evaluation (long run) |

From inside the container, with cwd `gym_dssat_pdi_samples`:

```bash
python3 -u 01_rule_based_baselines.py
python3 -u pc_env.py
python3 -u 03_sb3_sanity_check.py
python3 -u 04_pc_ppo_quick_train.py
```

---

## 5. Maintainer: push updates

Standard flow:

```bash
git status
git add -A
git commit -m "Describe change"
git push origin stable
```

Adjust **remote name** and **branch** if you use something other than `origin`/`stable`.

For **HTTPS** pushes, GitHub expects a **PAT**, not your account password. **SSH**:

```bash
git remote set-url origin git@github.com:firass-a/iconic.git
ssh -T git@github.com
git push origin stable
```

---

## Troubleshooting

- **`git push` → 401 in Cursor**: push from an external terminal, or fix GitHub credentials (PAT / `gh auth login` / SSH).
- **`pip` / NumPy conflicts in the container**: install SB3 deps **as root** and keep **`numpy==1.24.1`** and **`pandas<3`** as above.
- **Merge conflict markers** in Python files (e.g. `<<<<<<<`): resolve before sharing; corrupted files fail at import.

For general gym-DSSAT install and usage, see the [official documentation](https://rgautron.gitlabpages.inria.fr/gym-dssat-docs/Installation/index.html) linked from the main `README.md`.
