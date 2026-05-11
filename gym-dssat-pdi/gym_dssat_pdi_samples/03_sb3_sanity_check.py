"""
Step 2: SB3 sanity check for PCSmartFarmEnv.

Confirms that Stable-Baselines3's PPO can talk to our adapter without
crashing. Trains for 512 steps (no learning expected) just to validate
the plumbing end-to-end.

What this script verifies:
    1. check_env() passes — SB3's built-in Gymnasium API validator
    2. PPO can instantiate with MlpPolicy on our (15,) obs / Box(2,) action
    3. learn(total_timesteps=512) completes
    4. predict() works on a fresh obs
    5. save() / load() roundtrip works

What this script does NOT do:
    - Train enough to learn anything useful (real training is Step 3)
    - Measure reward trends or convergence
    - Evaluate the policy properly

Run inside Docker:
    pip install stable-baselines3        # if not already installed
    python3 /workspace/03_sb3_sanity_check.py
"""
import os
import sys
import numpy as np

# Force SB3 to use CPU. Gym-DSSAT is the bottleneck anyway (Fortran process
# in a subprocess), and putting tiny MLP forward passes on GPU adds latency
# without any throughput benefit.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
except ImportError as e:
    print("Missing dependency. Install with:")
    print("    pip install stable-baselines3")
    raise

# Local import — must live next to pc_env.py
from importlib import import_module
PCSmartFarmEnv = import_module('pc_env').PCSmartFarmEnv


def main():
    print("=" * 70)
    print("STEP 2 — SB3 SANITY CHECK")
    print("=" * 70)

    # ---- 1. Construct env ----
    print("\n[1] Building PCSmartFarmEnv...")
    env = PCSmartFarmEnv(mode='all', dssat_seed=123,
                         enable_faults=False, rng_seed=0)
    print(f"    action_space      = {env.action_space}")
    print(f"    observation_space = {env.observation_space}")

    # ---- 2. SB3 env validator ----
    # check_env runs a sequence of reset/step/space checks and raises if
    # anything violates the Gymnasium contract. The `warn=True` flag is
    # important: we want WARNINGS about non-standard behavior (e.g. obs
    # outside declared bounds) without them being fatal.
    print("\n[2] Running SB3 check_env()...")
    try:
        check_env(env, warn=True, skip_render_check=True)
        print("    ✓ check_env passed")
    except Exception as e:
        print(f"    ❌ check_env failed: {type(e).__name__}: {e}")
        env.close()
        sys.exit(1)

    # ---- 3. Instantiate PPO ----
    # n_steps=128 means PPO collects 128 transitions before each update.
    # For 512 total timesteps that's 4 update cycles — enough to confirm
    # the optimizer actually runs without throwing.
    print("\n[3] Instantiating PPO('MlpPolicy', env)...")
    model = PPO(
        policy='MlpPolicy',
        env=env,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        learning_rate=3e-4,
        verbose=1,
        device='cpu',
        seed=0,
    )
    print("    ✓ model built")
    print(f"    policy: {model.policy.__class__.__name__}")
    print(f"    device: {model.device}")

    # ---- 4. Train for 512 steps ----
    print("\n[4] Training for 512 timesteps (no learning expected)...")
    print("    [PPO output follows]")
    print("    " + "-" * 60)
    model.learn(total_timesteps=512, progress_bar=False)
    print("    " + "-" * 60)
    print("    ✓ learn() completed")

    # ---- 5. Inference on a fresh obs ----
    print("\n[5] Calling predict() on a fresh observation...")
    obs, _ = env.reset(seed=42)
    action, _ = model.predict(obs, deterministic=True)
    print(f"    obs shape      = {obs.shape}")
    print(f"    action shape   = {action.shape}")
    print(f"    action values  = {action}")
    assert action.shape == (2,), f"action shape wrong: {action.shape}"
    assert env.action_space.contains(action.astype(np.float32)), \
        f"predicted action {action} not in action_space"
    print("    ✓ predict OK and action within bounds")

    # ---- 6. Save / load roundtrip ----
    print("\n[6] Save / load roundtrip...")
    save_path = '/tmp/pcppo_sanity_check.zip'
    model.save(save_path)
    print(f"    saved to {save_path}")
    loaded = PPO.load(save_path, env=env, device='cpu')
    action_loaded, _ = loaded.predict(obs, deterministic=True)
    np.testing.assert_allclose(action, action_loaded, rtol=1e-5,
        err_msg="loaded model produced different action")
    print("    ✓ loaded model produces same action as in-memory model")

    env.close()
    print("\n" + "=" * 70)
    print("STEP 2 PASSED — pipeline is sound.")
    print("Ready for Step 3: 50k-step quick training run.")
    print("=" * 70)


if __name__ == '__main__':
    main()
