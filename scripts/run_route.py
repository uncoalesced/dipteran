"""Drive the full configured route, Point A to Point B, and report what happened.

A 12-second smoke test proves the loop turns over. It cannot show the things
that only appear over thousands of frames, and those are exactly the risks
PLAN.md section 11 leaves open for Phase 2:

  * stranding -- the OSMnx extract has 239 strongly-connected components, and
    a vehicle that wanders into a small one can get stuck there;
  * sustained off-network drift, which the soft edge-pull is supposed to
    prevent but has never been measured over a long run;
  * plastic weight divergence -- the three-factor rule fires on every tick
    with non-zero dopamine, so a slow drift compounds over tens of thousands
    of updates even though it looks harmless in a short run.

So this reports drift and geography, not just "it did not crash".

    python scripts/run_route.py --max-minutes 60

Engineered by uncoalesced.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ARRIVAL_M = 200.0        # close enough to call it arrival
TICKS_PER_FRAME = 5      # 5 x 20 ms = one 100 ms control frame
FRAME_DT = TICKS_PER_FRAME / 50.0


def run(cfg, max_minutes: float, progress_every: float, realtime: bool,
        origin: tuple[float, float] | None = None,
        goal: tuple[float, float] | None = None,
        max_episodes: int = 1, episode_minutes: float = 0.0) -> dict:
    import torch

    from dipteran.connectome import build_adjacency, resolve_clusters
    from dipteran.navigator import (FlyBangaloreNavigator, MotorOutput,
                                    classify_state, monoamine_rows)
    from dipteran.world import World

    # Connectome first, then SUMO: the build peak and the resident network do
    # not fit together, but the build peak is transient. See main.py.
    print("[route] building the connectome", flush=True)
    built = build_adjacency(cfg)
    clusters = resolve_clusters(cfg, neurons=built["neurons"])
    mono = monoamine_rows(cfg, built["neurons"])
    navigator = FlyBangaloreNavigator.from_config(
        cfg, built["W"], built["neurons"], clusters, monoamine=mono)
    del built

    print(f"[route] navigator ready, gain {cfg.snn.gain}", flush=True)
    print("[route] starting SUMO", flush=True)

    started = time.monotonic()
    deadline = started + max_minutes * 60.0
    next_report = started + progress_every
    # An episode ends on arrival or after this many frames, whichever comes
    # first. 0 means "no limit": one attempt that runs until it arrives or
    # the whole run times out, which is the pre-episode behaviour.
    episode_limit = int(episode_minutes * 60.0 / FRAME_DT)

    with World(cfg, step_length=cfg.sumo.step_length,
               origin=origin, goal=goal) as world:
        print(f"[route] {world.background} background vehicles", flush=True)
        s = world.state
        distance_start = s.distance_to_goal_m
        distance_min = distance_start

        frames = 0
        edges: set[str] = set()
        states: collections.Counter = collections.Counter()
        events: collections.Counter = collections.Counter()
        speeds: list[float] = []
        steers: list[float] = []
        errors: list[float] = []
        steer_votes = 0
        steer_agrees = 0
        # Episode bookkeeping -- see the episode boundary in the loop.
        episodes: list[dict] = []
        episode_frames = 0
        episode_start_m = distance_start
        episode_start_route_m = s.route_distance_to_goal_m
        route_min = s.route_distance_to_goal_m
        # Frames this episode spent with no live road route. Recorded because
        # a closure figure measured across a routing outage used to be a
        # different quantity from one measured without -- see
        # `World._route_distance_now`. Non-zero here means read the episode's
        # numbers with that in mind.
        episode_no_route_frames = 0
        off_network = 0
        off_road: list[float] = []
        panic_frames = 0
        pothole_frames = 0
        stall_peak = 0.0
        arrived = False
        # Measured from here, not from `started`: `started` precedes the
        # connectome build and SUMO's 75-130 s parse of the 862 MB network,
        # and folding startup into a steady-state rate understates it badly.
        loop_started = time.monotonic()
        # Per-phase timing. scripts/bench_sync.py measures a synthetic frame
        # and predicts ~16 frames/s; the real loop delivers far less, and the
        # only way to find out which call eats the difference is to time each
        # phase rather than reason about it.
        cost: collections.Counter = collections.Counter()

        while time.monotonic() < deadline:
            frame_started = time.perf_counter()

            t = time.perf_counter()
            sensory = world.sense(FRAME_DT)
            pam, ppl1 = world.rewards()
            sensory["reward_pam"] = pam
            sensory["punish_ppl1"] = ppl1
            navigator.set_sensory(sensory)
            cost["sense_ms"] += (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            # Averaged, not sampled: DNa02 is one neuron per side, so a
            # single tick quantises `steer` to multiples of 0.1.
            motor = MotorOutput.average(
                [navigator.step() for _ in range(TICKS_PER_FRAME)])
            cost["snn_ms"] += (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            world.apply(motor, FRAME_DT)
            cost["apply_ms"] += (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            world.step_sumo()
            cost["step_ms"] += (time.perf_counter() - t) * 1000
            frames += 1
            episode_frames += 1

            title, _ = classify_state(
                motor, speed_ms=s.speed, throttle_cmd=s.throttle_cmd,
                pothole=sensory["mechano_bristle"] > 0.0,
                horn=sensory["acoustic_jo"])
            states[title] += 1

            if s.edge:
                edges.add(s.edge)
            speeds.append(s.speed)
            steers.append(motor.steer)
            # The compass-versus-motor question, recorded per frame: how far
            # off the route target the fly is pointing, and whether it is
            # steering the way that error asks. A large standing error with
            # steer of the wrong sign is a different bug from the same error
            # with the right sign and not enough of it.
            errors.append(s.heading_error_deg)
            # Votes on the *filtered* command, because that is what the
            # vehicle turns on. Voting on the raw per-frame readout measures
            # the noise the filter exists to remove.
            command = s.steer_filtered
            if abs(s.heading_error_deg) > 10.0 and command != 0.0:
                steer_votes += 1
                if (command > 0) == (s.heading_error_deg > 0):
                    steer_agrees += 1
            if s.off_road_m == float("inf"):
                off_network += 1
            else:
                off_road.append(s.off_road_m)
            if motor.panic > 0.0:
                panic_frames += 1
            if sensory["mechano_bristle"] > 0.0:
                pothole_frames += 1
                events["pothole"] += 1
            if sensory["looming_lc4"] > 0.5:
                events["loom"] += 1
            stall_peak = max(stall_peak, s.stalled_s)
            distance_min = min(distance_min, s.distance_to_goal_m)
            route_min = min(route_min, s.route_distance_to_goal_m)
            if not world.route_valid:
                episode_no_route_frames += 1

            # -- episode boundary ------------------------------------------
            # Arriving, or running out of episode, ends an *attempt* rather
            # than the run. `reset_episode` teleports the fly back to the
            # origin and clears the navigator's membrane state, deliberately
            # keeping `p_val` -- the plastic overlay is the only thing the
            # fly carries forward, and it is the entire point of running
            # more than one attempt. At learning_rate 1e-4 a single pass is
            # far too short for the three-factor rule to move anything, so
            # until this loop existed every run was a naive fly.
            episode_over = (s.distance_to_goal_m <= ARRIVAL_M
                            or (episode_frames >= episode_limit > 0))
            if episode_over:
                hit = s.distance_to_goal_m <= ARRIVAL_M
                arrived = arrived or hit
                closed_route = episode_start_route_m - route_min
                # You cannot close more route than you drove. This is not a
                # style assertion: the 2026-09-19 twisty run reported closing
                # 1,862 m of route in an episode that travelled 790 m, and
                # that impossibility is the only reason the unit switch
                # behind it was ever found (see `World._route_distance_now`).
                # 1 m of slack absorbs the rounding in `travelled_m`, which
                # accumulates per-frame haversine steps.
                assert closed_route <= s.travelled_m + 1.0, (
                    f"episode {len(episodes) + 1} closed {closed_route:.1f} m "
                    f"of route on {s.travelled_m:.1f} m travelled -- route "
                    f"distance changed units or jumped; "
                    f"{episode_no_route_frames} frames had no live route")
                episodes.append({
                    "episode": len(episodes) + 1,
                    "arrived": hit,
                    "frames": episode_frames,
                    "closed_m": round(episode_start_m - distance_min, 1),
                    # Crow-flies closure badly understates a one-way system:
                    # this network has pairs 500 m apart that are a 2.4 km
                    # drive, where real progress barely moves the straight
                    # line. Route distance is what the fly is actually
                    # working against, and what `rewards()` gates on.
                    "closed_route_m": round(closed_route, 1),
                    "best_route_m": round(route_min, 1),
                    "best_distance_m": round(distance_min, 1),
                    "travelled_m": round(s.travelled_m, 1),
                    # Frames with no live road route. Above zero means the
                    # route figures were held rather than measured for part
                    # of this episode.
                    "no_route_frames": episode_no_route_frames,
                    "plastic_drift": round(navigator.drift(), 6),
                })
                print(f"[route] episode {len(episodes)}: "
                      f"{'ARRIVED' if hit else 'timed out'} after "
                      f"{episode_frames} frames, closed "
                      f"{closed_route:.0f} m of route "
                      f"({episode_start_m - distance_min:.0f} m direct), "
                      f"travelled {s.travelled_m:.0f} m, "
                      f"{episode_no_route_frames} frames no route, "
                      f"drift {navigator.drift():.2e}", flush=True)
                if len(episodes) >= max_episodes:
                    break
                world.reset_episode(navigator)
                episode_frames = 0
                episode_start_m = s.distance_to_goal_m
                distance_min = s.distance_to_goal_m
                episode_start_route_m = s.route_distance_to_goal_m
                route_min = s.route_distance_to_goal_m
                episode_no_route_frames = 0
                continue

            now = time.monotonic()
            if now >= next_report:
                next_report = now + progress_every
                drift = navigator.drift()
                print(f"[route] {frames:>6} frames  sim {frames * FRAME_DT:7.0f}s  "
                      f"to goal {s.distance_to_goal_m / 1000:6.2f} km  "
                      f"travelled {s.travelled_m / 1000:5.2f} km  "
                      f"speed {s.speed:5.2f} m/s  edges {len(edges):>4}  "
                      f"drift {drift:.2e}", flush=True)

            if realtime:
                slack = FRAME_DT - (time.perf_counter() - frame_started)
                if slack > 0:
                    time.sleep(slack)

        wall = time.monotonic() - started
        loop_wall = time.monotonic() - loop_started
        sim_s = frames * FRAME_DT
        drift = navigator.drift()

        return {
            "arrived": arrived,
            "episodes": episodes,
            "episode_count": len(episodes),
            "episodes_arrived": sum(1 for e in episodes if e["arrived"]),
            # The number that says whether anything was learned: closed
            # distance in the last third of the episodes against the first.
            "closed_m_first_third": _mean_closed(episodes, "first"),
            "closed_m_last_third": _mean_closed(episodes, "last"),
            "origin": [round(world.origin[0], 6), round(world.origin[1], 6)],
            "goal": [round(world.goal[0], 6), round(world.goal[1], 6)],
            "frames": frames,
            "sim_seconds": round(sim_s, 1),
            "wall_seconds": round(wall, 1),
            "startup_seconds": round(loop_started - started, 1),
            "loop_seconds": round(loop_wall, 1),
            "realtime_factor": (round(sim_s / loop_wall, 2)
                                if loop_wall else None),
            "realtime_paced": realtime,
            "distance_start_m": round(distance_start, 1),
            "distance_end_m": round(s.distance_to_goal_m, 1),
            "distance_min_m": round(distance_min, 1),
            "closed_m": round(distance_start - distance_min, 1),
            "travelled_m": round(s.travelled_m, 1),
            "edges_visited": len(edges),
            "final_edge": s.edge,
            "off_network_frames": off_network,
            "off_road_m_median": (round(statistics.median(off_road), 2)
                                  if off_road else None),
            "off_road_m_max": round(max(off_road), 2) if off_road else None,
            "speed_ms_median": (round(statistics.median(speeds), 3)
                                if speeds else None),
            "speed_ms_max": round(max(speeds), 3) if speeds else None,
            "steer_abs_median": (round(statistics.median(abs(v) for v in steers), 4)
                                 if steers else None),
            "steer_abs_max": (round(max(abs(v) for v in steers), 4)
                              if steers else None),
            # Is the fly pointing where the road goes, and does it steer the
            # way the error asks when it is not?
            "heading_error_abs_median": (
                round(statistics.median(abs(e) for e in errors), 1)
                if errors else None),
            "heading_error_abs_p90": (
                round(sorted(abs(e) for e in errors)[int(len(errors) * 0.9)], 1)
                if errors else None),
            "steer_sign_agreement": (round(steer_agrees / steer_votes, 3)
                                     if steer_votes else None),
            "steer_sign_votes": steer_votes,
            "panic_frames": panic_frames,
            "pothole_frames": pothole_frames,
            "stall_peak_s": round(stall_peak, 1),
            "events": dict(events),
            "states": dict(states),
            # The number a short run cannot produce: how far reinforcement has
            # moved the weights from the anatomy, relative to the anatomy's
            # own magnitude. Bounded by design, so a large value here means
            # the bound is wrong, not that the fly learned a lot.
            "plastic_drift_relative": round(drift, 6),
            "plastic_finite": bool(torch.isfinite(navigator.p_val).all()),
            "plastic_synapses": navigator.plastic_synapses,
            "frames_per_s": (round(frames / loop_wall, 2)
                             if loop_wall else None),
            "phase_ms_per_frame": {k: round(v / max(frames, 1), 2)
                                   for k, v in cost.items()},
            "background_vehicles": world.background,
            "min_syn_count": cfg.compute.min_syn_count,
            "gain": cfg.snn.gain,
        }


def _mean_closed(episodes: list[dict], which: str) -> float | None:
    """Mean distance closed over the first or last third of the episodes.

    Learning, if it happens at all here, is slow and noisy -- comparing
    single episodes would mostly report traffic. Thirds are the smallest
    summary that can show a trend without pretending to a precision a
    handful of runs does not have.
    """
    if len(episodes) < 3:
        return None
    cut = max(1, len(episodes) // 3)
    chunk = episodes[:cut] if which == "first" else episodes[-cut:]
    return round(sum(e["closed_m"] for e in chunk) / len(chunk), 1)


def _latlon(text: str) -> tuple[float, float]:
    """"12.9177,77.6238" -> (12.9177, 77.6238)."""
    lat, _, lon = text.partition(",")
    try:
        return float(lat), float(lon)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected LAT,LON (e.g. 12.9177,77.6238), got {text!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-minutes", type=float, default=60.0)
    ap.add_argument("--progress-every", type=float, default=60.0,
                    help="seconds between progress lines")
    ap.add_argument("--realtime", action="store_true",
                    help="pace to wall clock; off by default so a long route "
                         "finishes in the time the hardware allows")
    ap.add_argument("--out", type=Path, default=None)
    # Overrides for testing one kind of road at a time -- a straight run, a
    # single turn, a street grid -- without editing config.yaml between
    # runs. Omit both and the configured route stands.
    ap.add_argument("--origin", type=_latlon, default=None,
                    metavar="LAT,LON", help="override the route origin")
    ap.add_argument("--dest", type=_latlon, default=None,
                    metavar="LAT,LON", help="override the destination")
    # Episodic training. The plastic overlay survives `reset_episode`, so
    # repeated attempts at the same route are the only way the three-factor
    # rule gets enough updates to move anything at learning_rate 1e-4.
    ap.add_argument("--episodes", type=int, default=1,
                    help="attempts at the route, weights carried between them")
    ap.add_argument("--episode-minutes", type=float, default=0.0,
                    help="cap on one attempt; 0 means run until arrival")
    args = ap.parse_args()

    from dipteran.config import load_config

    cfg = load_config()
    result = run(cfg, args.max_minutes, args.progress_every, args.realtime,
                 origin=args.origin, goal=args.dest,
                 max_episodes=max(1, args.episodes),
                 episode_minutes=args.episode_minutes)
    print("\n" + json.dumps(result, indent=2), flush=True)

    out = args.out or (cfg.paths.cache_dir / "route_run.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[route] wrote {out}", flush=True)

    if not result["plastic_finite"]:
        print("FAIL: plastic weights went non-finite", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
