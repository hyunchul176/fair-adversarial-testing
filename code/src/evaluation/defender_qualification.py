"""Shared fixed-seed normal-driving qualification for defender policies."""
from __future__ import annotations

import numpy as np


def evaluate_policy(model, env_factory, *, seed_base: int, episodes: int, max_steps: int,
                    qualification: dict):
    env = env_factory()
    rows = []
    try:
        for episode in range(episodes):
            environment_seed = seed_base + episode
            observation, _ = env.reset(seed=environment_seed)
            collision = offroad = wrong_way = False
            speeds = []
            interventions = 0
            info = {}
            for step in range(max_steps):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                collision |= bool(info["collision"])
                offroad |= not bool(info["body_onroad"])
                wrong_way |= bool(info["wrong_way"])
                interventions += int(info.get("shield_intervened", False))
                speeds.append(float(info["forward_speed"]))
                if terminated or truncated:
                    break
            success = bool(
                not collision and not offroad and not wrong_way and step + 1 == max_steps
                and float(info["progress_m"]) >= 300.0
            )
            rows.append({
                "episode": episode,
                "environment_seed": environment_seed,
                "steps": step + 1,
                "collision": collision,
                "offroad": offroad,
                "wrong_way": wrong_way,
                "success": success,
                "progress_m": float(info["progress_m"]),
                "mean_forward_speed": float(np.mean(speeds)),
                "intervention_fraction": interventions / (step + 1),
                "termination_reason": info["termination_reason"],
            })
    finally:
        env.close()
    summary = {
        "episodes": len(rows),
        "collision_pct": float(100 * np.mean([row["collision"] for row in rows])),
        "offroad_pct": float(100 * np.mean([row["offroad"] for row in rows])),
        "wrong_way_pct": float(100 * np.mean([row["wrong_way"] for row in rows])),
        "success_pct": float(100 * np.mean([row["success"] for row in rows])),
        "mean_forward_speed": float(np.mean([row["mean_forward_speed"] for row in rows])),
        "mean_progress_m": float(np.mean([row["progress_m"] for row in rows])),
        "mean_intervention_fraction": float(np.mean([row["intervention_fraction"] for row in rows])),
    }
    checks = {
        "collision": summary["collision_pct"] <= float(qualification["max_collision_pct"]),
        "offroad": summary["offroad_pct"] <= float(qualification["max_offroad_pct"]),
        "wrong_way": summary["wrong_way_pct"] <= float(qualification["max_wrong_way_pct"]),
        "success": summary["success_pct"] >= float(qualification["min_success_pct"]),
        "speed": summary["mean_forward_speed"] >= float(qualification["min_mean_forward_speed"]),
    }
    summary.update(checks={key: bool(value) for key, value in checks.items()},
                   qualified=bool(all(checks.values())))
    return rows, summary
