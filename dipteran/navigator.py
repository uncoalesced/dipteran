"""FlyBangaloreNavigator -- the real SNN loop over the FlyWire connectome.

This replaces PLAN.md section 5's reference sketch, which was written before
anything had been measured. Three of its assumptions do not survive contact
with the real matrix, and each is corrected here:

1.  `v_thresh = 1.0` against raw weights. Edge weights are aggregate synapse
    counts running to 2,405 and in-degree runs to hundreds, so a single
    presynaptic spike can carry a neuron thousands of times past a threshold
    of 1. Rows are normalised by their own total input weight, which turns
    synaptic drive into "what fraction of my input is firing" and makes one
    global `gain` the only excitability knob left. Calibrate it with
    `calibrate_gain()` -- do not guess it.

2.  Dopamine as a fast excitatory synapse. `NT_SIGN` signs dopamine,
    serotonin and octopamine +1 as a placeholder, which PLAN.md section 11
    flags as wrong biology. Monoamines are neuromodulators: they do not drive
    their targets on a 20 ms timescale, they change how much other synapses
    count for. Their columns are dropped from the fast current path here, and
    dopamine instead acts on weights through the plastic overlay below.

3.  `self.v[spikes.bool()] = self.v_reset` inside a Python dict loop over
    external currents. Both are rewritten as whole-tensor ops -- the loop was
    per-neuron and would dominate the tick.

The reinforcement rule is the three-factor one PLAN.md section 4 asks for:
an eligibility trace of recent (motor, presynaptic) coincidence, gated by the
PAM-minus-PPL1 dopamine signal. No backpropagation anywhere.

Engineered by uncoalesced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Bangalore traffic states, keyed off what the decoded motors are doing.
# The HUD design borrowed its *layout* from a third-party fly-brain visualiser
# -- point cloud, state banner, per-region legend -- and nothing else. That
# visualiser reacted to social-media behaviour; these captions are PLAN.md's
# traffic states instead. No emojis, by standing decision (PLAN.md section 9).
STATES = (
    ("PANIC SWERVE",        "Giant fiber fired. Evasive, uncontrolled."),
    ("POTHOLE IMPACT",      "Campaniform shock. Motor output frozen."),
    ("HORN OVERDRIVE",      "Johnston's organ saturated. Compass destabilised."),
    ("GRIDLOCK STAGNATION", "Throttle open, zero optic flow. Trapped."),
    ("ODOR TRACKING",       "Climbing the gradient toward Point B."),
    ("FULL THROTTLE",       "Clean air, clean road. Wingbeat at amplitude."),
    ("IDLE DRIFT",          "No dominant drive. Wandering."),
)


@dataclass
class MotorOutput:
    """One tick of decoded motor state, all in [0, 1] except steer."""

    steer: float = 0.0          # -1 hard left .. +1 hard right (DNa02 R-L)
    throttle: float = 0.0       # DNg02 wingbeat amplitude
    panic: float = 0.0          # DNp01 giant fiber; overrides throttle
    dopamine: float = 0.0       # PAM - PPL1, the reinforcement signal
    reward: float = 0.0         # PAM alone
    punishment: float = 0.0     # PPL1 alone
    spike_rate: float = 0.0     # whole-population mean, for calibration
    regions: dict = field(default_factory=dict)   # per-cluster activation

    @classmethod
    def average(cls, outputs: "list[MotorOutput]") -> "MotorOutput":
        """Mean of a frame's ticks, except `panic`, which is a maximum.

        The steering clusters are *one neuron per side* -- DNa02 is a single
        bilateral pair -- so a single tick's rate can only ever be a multiple
        of `1/subticks`, i.e. 0.1. Reading one tick per frame therefore
        quantises `steer` to about three usable levels, and none of them is
        zero once the resting bias is subtracted. Averaging the frame's ticks
        samples `subticks * len(outputs)` sub-steps instead and restores the
        resolution the difference needs.

        `panic` is maxed rather than averaged: a giant-fiber spike anywhere in
        the frame is an escape, and averaging it away would silently disable
        the one reflex that is supposed to override everything.
        """
        if not outputs:
            return cls()
        n = float(len(outputs))
        regions: dict[str, float] = {}
        for out in outputs:
            for name, value in out.regions.items():
                regions[name] = regions.get(name, 0.0) + value
        return cls(
            steer=sum(o.steer for o in outputs) / n,
            throttle=sum(o.throttle for o in outputs) / n,
            panic=max(o.panic for o in outputs),
            dopamine=sum(o.dopamine for o in outputs) / n,
            reward=sum(o.reward for o in outputs) / n,
            punishment=sum(o.punishment for o in outputs) / n,
            spike_rate=sum(o.spike_rate for o in outputs) / n,
            regions={k: v / n for k, v in regions.items()},
        )


class FlyBangaloreNavigator:
    """The connectome as a controller.

    `w` is the signed sparse CSR from `connectome.build_adjacency`, `neurons`
    its row order, `clusters` the payload from `connectome.resolve_clusters`.
    """

    def __init__(self, w, neurons, clusters, *, gain: float = 1.0,
                 decay: float = 0.9, v_thresh: float = 1.0,
                 v_reset: float = -0.2, subticks: int = 10,
                 learning_rate: float = 0.01, trace_decay: float = 0.8,
                 sensory_gain: float = 1.0, homeostasis: float = 0.001,
                 plastic_span: float = 0.5, steer_bias: float = 0.0,
                 monoamine_rows: np.ndarray | None = None):
        import torch

        self.torch = torch
        self.n = w.shape[0]
        self.device = w.device
        self.dtype = w.values().dtype

        self.decay = decay
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.subticks = subticks
        self.gain = gain
        self.sensory_gain = sensory_gain
        self.learning_rate = learning_rate
        self.trace_decay = trace_decay
        self.homeostasis = homeostasis
        self.plastic_span = plastic_span
        self.steer_bias = steer_bias

        self.neurons = neurons
        self.rows = {name: torch.as_tensor(payload["rows"], dtype=torch.long,
                                           device=self.device)
                     for name, payload in clusters["clusters"].items()}
        self.roles = {name: payload["role"]
                      for name, payload in clusters["clusters"].items()}

        self.w = self._prepare(w, monoamine_rows)

        self.v = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        self.spikes = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        self.i_ext = torch.zeros(self.n, device=self.device, dtype=self.dtype)

        # Plasticity is restricted to the synapses onto the motor clusters.
        # ponytail: biologically the modulated site is Kenyon cell -> MBON
        # inside the mushroom body, not the descending motor neurons. Doing it
        # properly means resolving the KC and MBON populations and keeping a
        # third matrix; this puts the same three-factor rule on the readout
        # the vehicle actually obeys, which is where a wrong turn has to
        # become less likely. Move it to KC->MBON if the learned behaviour
        # turns out to depend on where the plasticity lives.
        #
        # Stored sparsely, as the synapses that actually exist. The dense
        # (n_motor x n) form is the obvious one and it is a trap: here that is
        # 356 x 138,639 floats, 198 MB, of which ~99.9% are structural zeros,
        # because mean in-degree in this connectome is about 109. Reading it
        # ten times a tick cost 364 ms per frame against a 57 ms budget. The
        # sparse form is ~39k entries and makes the same arithmetic free.
        motor_names = [n for n, role in self.roles.items() if role == "motor"]
        self.motor_rows = torch.cat([self.rows[n] for n in motor_names])
        self.n_motor = len(self.motor_rows)

        indptr = self.w.crow_indices()
        cols, vals = self.w.col_indices(), self.w.values()
        p_row, p_col, p_val = [], [], []
        for i, row in enumerate(self.motor_rows.tolist()):
            lo, hi = int(indptr[row]), int(indptr[row + 1])
            if hi <= lo:
                continue
            p_row.append(torch.full((hi - lo,), i, dtype=torch.long,
                                    device=self.device))
            p_col.append(cols[lo:hi].long())
            p_val.append(vals[lo:hi])

        if p_row:
            self.p_row = torch.cat(p_row)
            self.p_col = torch.cat(p_col)
            self.p_val = torch.cat(p_val).clone()
        else:
            empty_i = torch.zeros(0, dtype=torch.long, device=self.device)
            self.p_row, self.p_col = empty_i, empty_i
            self.p_val = torch.zeros(0, device=self.device, dtype=self.dtype)

        self.p_val0 = self.p_val.clone()           # anatomy, for bounding drift
        self.p_abs0 = self.p_val0.abs()
        self.trace = torch.zeros_like(self.p_val)
        self.plastic_synapses = int(self.p_val.numel())

    # -- setup -----------------------------------------------------------

    def _prepare(self, w, monoamine_rows):
        """Drop monoamine columns, then row-normalise by total input weight."""
        torch = self.torch
        indptr, indices = w.crow_indices(), w.col_indices()
        values = w.values().clone()

        if monoamine_rows is not None and len(monoamine_rows):
            mask = torch.zeros(self.n, dtype=torch.bool, device=self.device)
            mask[torch.as_tensor(monoamine_rows, dtype=torch.long,
                                 device=self.device)] = True
            values[mask[indices.long()]] = 0.0
            self.monoamine_count = int(mask.sum())
        else:
            self.monoamine_count = 0

        counts = (indptr[1:] - indptr[:-1]).long()
        row_of = torch.repeat_interleave(
            torch.arange(self.n, device=self.device), counts)
        totals = torch.zeros(self.n, device=self.device, dtype=self.dtype)
        totals.scatter_add_(0, row_of, values.abs())
        # A row with no surviving input keeps a denominator of 1 rather than
        # dividing by zero; its values are all 0 anyway.
        values = values / totals[row_of].clamp(min=1.0)

        self.in_degree = counts
        self.input_weight = totals
        return torch.sparse_csr_tensor(indptr, indices, values,
                                       size=w.shape, device=self.device)

    def _dense_rows(self, w, rows):
        """Pull a handful of CSR rows out as a small dense matrix."""
        torch = self.torch
        indptr = w.crow_indices()
        indices, values = w.col_indices(), w.values()
        out = torch.zeros((len(rows), self.n), device=self.device,
                          dtype=self.dtype)
        for i, row in enumerate(rows.tolist()):
            lo, hi = int(indptr[row]), int(indptr[row + 1])
            out[i, indices[lo:hi].long()] = values[lo:hi]
        return out

    # -- the loop --------------------------------------------------------

    def set_sensory(self, currents: dict[str, float]) -> None:
        """Inject per-cluster external current. Keys are cluster names.

        `sensory_gain` is a separate knob from `gain` on purpose. `gain`
        scales the *recurrent* drive and was calibrated against a uniform
        stimulus; the world does not supply a uniform stimulus. An inverse
        odour law is 0.047 at the 10.6 km start of this route -- physically
        right, and nowhere near a 1.0 threshold -- so without an input scale
        the network is correctly tuned and still barely driven.
        """
        self.i_ext.zero_()
        for name, value in currents.items():
            rows = self.rows.get(name)
            if rows is None or not value:
                continue
            self.i_ext[rows] = float(value) * self.sensory_gain

    def step(self) -> MotorOutput:
        """One 20 ms tick: `subticks` LIF sub-steps, then decode the motors."""
        torch = self.torch
        fired = torch.zeros(self.n, device=self.device, dtype=self.dtype)

        for _ in range(self.subticks):
            drive = torch.mv(self.w, self.spikes) * self.gain
            # The motor rows read from the plastic overlay instead, so that
            # dopamine changes what reaches them. Everything else is W.
            motor_drive = torch.zeros(self.n_motor, device=self.device,
                                      dtype=self.dtype)
            motor_drive.index_add_(0, self.p_row,
                                   self.p_val * self.spikes[self.p_col])
            drive[self.motor_rows] = motor_drive * self.gain

            self.v = self.v * self.decay + drive + self.i_ext
            self.spikes = (self.v >= self.v_thresh).to(self.dtype)
            self.v = torch.where(self.spikes.bool(),
                                 torch.full_like(self.v, self.v_reset), self.v)
            fired += self.spikes

        rates = fired / self.subticks

        # Eligibility trace: which (motor, presynaptic) pairs co-fired
        # recently. Decays on its own, so only recent coincidence is still
        # eligible when dopamine arrives.
        #
        # Maintained once per *tick*, not once per sub-tick, and that is a
        # measured decision rather than a shortcut. `trace` and `plastic` are
        # dense (n_motor x n) tensors -- 198 MB each here -- so the naive
        # placement inside the sub-tick loop touches ~1.5 G elements per tick
        # and swamps the sparse matmul entirely: profiling the real loop gave
        # 364 ms per frame against the 57 ms `scripts/bench_sync.py` predicts,
        # and the whole difference was these two lines. Tick-level rates are
        # the right resolution anyway, because `learning_rate` is 1e-4 and the
        # rule is designed to need ~1000 ticks to move anything; sub-tick
        # precision in the trace is detail the update can never act on.
        self.trace *= self.trace_decay
        self.trace += rates[self.motor_rows][self.p_row] * rates[self.p_col]

        out = self._decode(rates)
        self._reinforce(out.dopamine)
        return out

    def _decode(self, rates) -> MotorOutput:
        def rate(name: str) -> float:
            rows = self.rows.get(name)
            if rows is None or not len(rows):
                return 0.0
            return float(rates[rows].mean())

        reward = rate("reward_pam")
        punishment = rate("punish_ppl1")
        panic = rate("panic_dnp01")
        throttle = rate("throttle")

        # Steering is the right-minus-left difference *within* DNa02, which is
        # the quantity the literature relates to rotational velocity, minus a
        # calibrated resting bias.
        #
        # The bias term is not a fudge. Measured on this connectome, DNa02-left
        # is silent (0.000 Hz) while DNa02-right runs at ~101 Hz under
        # information-free drive, and that is not a reconstruction gap -- the
        # two cells have in-degree 742 and 751, input weight 12,237 and 13,116.
        # The asymmetry is the recurrent network's own excitation/inhibition
        # balance. Left raw, `steer` is therefore positive on every tick and
        # the vehicle can only ever turn right, which is an orbit. Subtracting
        # the resting value re-centres the readout so the PFL3 drive can push
        # it either way. `snn.steer_bias` is measured by
        # `scripts/probe_diagnostics.py`, not guessed -- same discipline as
        # `snn.gain`.
        steer = rate("steer_right") - rate("steer_left") - self.steer_bias

        return MotorOutput(
            steer=steer,
            throttle=0.0 if panic > 0.0 else throttle,
            panic=panic,
            dopamine=reward - punishment,
            reward=reward,
            punishment=punishment,
            spike_rate=float(rates.mean()),
            regions={name: rate(name) for name in self.rows},
        )

    def _reinforce(self, dopamine: float) -> None:
        """Three-factor update: eligibility x dopamine, no gradients.

        PAM firing potentiates whatever the motor neurons were just doing;
        PPL1 depresses it. This is the only learning in the system.

        The update is *multiplicative* -- scaled by each synapse's own
        anatomical weight -- and that is not a stylistic choice. `plastic`
        holds row-normalised weights, so a typical entry is around 1e-4,
        while the eligibility trace saturates at `1 / (1 - trace_decay)` = 5.
        An additive `learning_rate * dopamine * trace` is therefore about
        0.05 against a synapse of 1e-4: five hundred times the thing it is
        modifying. A 55-minute route run found exactly that -- relative drift
        went 9.6e-03 -> 6.0e-01 within 40 seconds, pinned against the clamp,
        and the motor readout collapsed to a permanent 0.00 m/s stall.
        Scaling by |plastic0| also gets the biology right: you cannot
        strengthen a synapse that does not exist.

        Homeostasis pulls slowly back toward the anatomy so a bad episode is
        survivable. Without it a sustained punishment -- a long stall, which
        is common in Bangalore traffic -- depresses the very motor pathway
        the fly needs in order to escape, and the stall becomes permanent.
        """
        # Homeostatic return runs every tick, dopamine or not; recovery from
        # a punished episode cannot depend on being rewarded later.
        if self.homeostasis:
            self.p_val += self.homeostasis * (self.p_val0 - self.p_val)

        if dopamine:
            self.p_val += (self.learning_rate * dopamine * self.trace
                           * self.p_abs0)

        # Keep learned weights in a bounded band around the anatomy, so
        # reinforcement can bias the connectome but never invent a pathway
        # that is not in it, and never flip an inhibitory synapse excitatory.
        # Symmetric about the anatomy, which a `min=w*0.2, max=w*2.5` form is
        # not: for an inhibitory synapse that inverts, min lands above max,
        # and torch silently returns the max.
        span = self.p_abs0 * self.plastic_span
        self.p_val = self.torch.clamp(self.p_val, min=self.p_val0 - span,
                                      max=self.p_val0 + span)

    def drift(self) -> float:
        """Relative distance of the learned weights from the anatomy.

        Bounded by `plastic_span` by construction, so a value near that bound
        means the reinforcement rule is saturating rather than that the fly
        learned a great deal.
        """
        anatomy = float(self.torch.linalg.norm(self.p_val0))
        moved = float(self.torch.linalg.norm(self.p_val - self.p_val0))
        return moved / max(anatomy, 1e-9)

    def reset(self) -> None:
        self.v.zero_()
        self.spikes.zero_()
        self.trace.zero_()

    @classmethod
    def from_config(cls, cfg, w, neurons, clusters,
                    monoamine: np.ndarray | None = None):
        """Build with the calibrated parameters from config.yaml."""
        s = cfg.snn
        return cls(w, neurons, clusters, gain=s.gain, decay=s.decay,
                   v_thresh=s.v_thresh, v_reset=s.v_reset,
                   subticks=s.subticks, learning_rate=s.learning_rate,
                   trace_decay=s.trace_decay, sensory_gain=s.sensory_gain,
                   homeostasis=s.homeostasis, plastic_span=s.plastic_span,
                   steer_bias=s.steer_bias, monoamine_rows=monoamine)


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def monoamine_rows(cfg, neurons: np.ndarray) -> np.ndarray:
    """Matrix rows whose neuron releases a monoamine.

    These are the ones PLAN.md section 11 flags as wrongly signed excitatory.
    They are dropped from the fast path and act through plasticity instead.
    """
    import pandas as pd

    from dipteran.connectome import load_annotations

    frame, _ = load_annotations(cfg)
    nt = frame["top_nt"].fillna("").str.strip().str.lower()
    mono = frame.loc[nt.isin({"dopamine", "serotonin", "octopamine"}), "root_id"]
    ids = pd.to_numeric(mono, errors="coerce").dropna().astype(np.int64).to_numpy()
    ids = np.unique(ids)

    positions = np.searchsorted(neurons, ids)
    positions = np.clip(positions, 0, len(neurons) - 1)
    return positions[neurons[positions] == ids]


# Measured floor, not a preference -- see the selection rule in
# calibrate_gain. Below this the per-tick motor rate estimate is zero almost
# every tick and the vehicle cannot steer.
MOTOR_FLOOR_HZ = 10.0

# The band a *behaving* fly's brain occupies. A resting brain sits nearer
# 1-20 Hz, but this one is under continuous sensory drive in traffic, and the
# ceiling here is set to leave room for that rather than to flatter a number.
TARGET_HZ = (1.0, 50.0)


def calibrate_gain(w, neurons, clusters, *, targets=TARGET_HZ,
                   gains=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0),
                   ticks: int = 20, drive: float = 0.4,
                   monoamine: np.ndarray | None = None) -> dict:
    """Sweep `gain` and report the firing rates each one produces.

    PLAN.md asks Phase 2 to "calibrate LIF parameters against observed spike
    rates" -- this is that, run rather than assumed. A sub-tick is 2 ms, so
    rate_hz = spikes per sub-tick * 500.

    Three rates, because the whole-population mean is misleading on its own.
    Sensory rows carry a clamped `i_ext`, so they fire at whatever rate the
    clamp dictates no matter what `gain` is; averaging them in measures the
    stimulus rather than the network. `network_hz` therefore excludes every
    clamped row, and `motor_hz` is reported separately because it is the one
    that has to be non-zero for the vehicle to move at all -- a beautifully
    calibrated network whose descending neurons never spike is a parked car.

    `drive` is deliberately below `v_thresh`: a sensory current at threshold
    makes those neurons fire every single sub-tick (500 Hz), which is not a
    stimulus, it is a short circuit.
    """
    import torch

    measured = []
    for gain in gains:
        nav = FlyBangaloreNavigator(w, neurons, clusters, gain=gain,
                                    monoamine_rows=monoamine)
        sensory = [name for name, role in nav.roles.items() if role == "sensory"]
        nav.set_sensory({name: drive for name in sensory})

        clamped = torch.zeros(nav.n, dtype=torch.bool, device=nav.device)
        for name in sensory:
            clamped[nav.rows[name]] = True
        free = ~clamped

        alls, nets, motors = [], [], []
        for _ in range(ticks):
            out = nav.step()
            alls.append(out.spike_rate)
            nets.append(float(nav.spikes[free].mean()))
            motors.append(float(nav.spikes[nav.motor_rows].mean()))

        # Discard the first third as transient; the rest is the steady state.
        cut = ticks // 3
        net_rate = float(np.mean(nets[cut:]))
        motor_rate = float(np.mean(motors[cut:]))
        measured.append({
            "gain": gain,
            "population_hz": round(float(np.mean(alls[cut:])) * 500.0, 2),
            "network_hz": round(net_rate * 500.0, 2),
            "motor_hz": round(motor_rate * 500.0, 2),
            "silent": net_rate == 0.0,
            "motor_silent": motor_rate == 0.0,
            "saturated": net_rate > 0.5,
        })
        del nav

    lo, hi = targets
    # Two constraints, and the second is an engineering one rather than a
    # biological preference. The tick samples `subticks` sub-steps at 2 ms, so
    # a motor population firing at r Hz contributes r/50 spikes to a tick's
    # rate estimate. Below about 10 Hz that estimate is zero almost every
    # tick, the steering difference is identically zero, and the vehicle does
    # not turn -- no matter how well the rest of the network is tuned. So a
    # gain is only usable if it clears MOTOR_FLOOR_HZ as well as keeping the
    # network off both rails.
    usable = [m for m in measured
              if lo <= m["network_hz"] <= hi
              and m["motor_hz"] >= MOTOR_FLOOR_HZ]
    # Prefer the *lowest* usable gain: gain scales the anatomy, so the
    # smallest one that produces a working controller is the one that
    # distorts the connectome least.
    best = min(usable, key=lambda m: m["gain"]) if usable else None

    return {
        "target_hz": list(targets),
        "motor_floor_hz": MOTOR_FLOOR_HZ,
        "sensory_drive": drive,
        "measurements": measured,
        "recommended_gain": best["gain"] if best else None,
        "recommended_network_hz": best["network_hz"] if best else None,
        "recommended_motor_hz": best["motor_hz"] if best else None,
    }


# --------------------------------------------------------------------------
# telemetry
# --------------------------------------------------------------------------

def classify_state(out: MotorOutput, *, speed_ms: float, throttle_cmd: float,
                   pothole: bool = False, horn: float = 0.0) -> tuple[str, str]:
    """(title, sub-caption) for the HUD banner.

    Order matters: the earlier a branch sits, the more it overrides. A giant
    fiber spike beats everything, because biologically it does.
    """
    if out.panic > 0.0:
        return STATES[0]
    if pothole:
        return STATES[1]
    if horn > 0.5:
        return STATES[2]
    if throttle_cmd > 0.3 and speed_ms < 0.5:
        return STATES[3]
    if out.reward > out.punishment and out.reward > 0.0:
        return STATES[4]
    if out.throttle > 0.5:
        return STATES[5]
    return STATES[6]


def telemetry(out: MotorOutput, *, lat: float, lon: float, heading: float,
              speed_ms: float, throttle_cmd: float, edge: str = "",
              nearby_agents=None, pothole: bool = False,
              horn: float = 0.0, event: str | None = None,
              fov_deg: float = 120.0, fov_radius_m: float = 60.0,
              heading_error_deg: float = 0.0, wrong_way_ms: float = 0.0,
              speed_cap_ms: float = 0.0, steer_filtered: float = 0.0) -> dict:
    """The JSON payload schema the HUD consumes.

    Extends the mock's fields rather than replacing them, so index.html's
    existing bindings keep working: lat/lon/heading/sugar_reward/pain_level/
    giant_fiber/event are all still here and still mean the same thing.

    `fov_deg`/`fov_radius_m` describe the sensing cone the HUD draws in
    front of the fly (Joel's "haze and highlight to show what the fly
    sees"). Defaults are world.py's LOOM_FIELD_DEG doubled to a full angle
    and LOOM_RADIUS_M -- the actual window `world.py._perceive` gates the
    looming/horn channels on -- but main.py always passes the real values
    explicitly so this is the fallback for any other caller, not the source
    of truth.
    """
    title, caption = classify_state(out, speed_ms=speed_ms,
                                    throttle_cmd=throttle_cmd,
                                    pothole=pothole, horn=horn)
    return {
        # -- the mock's original fields, now real ------------------------
        "lat": lat,
        "lon": lon,
        "heading": heading,
        "sugar_reward": round(out.reward, 4),
        "pain_level": round(out.punishment, 4),
        "giant_fiber": round(out.panic, 4),
        "event": event,
        # -- Phase 2 additions -------------------------------------------
        "state": title,
        "caption": caption,
        "steer": round(out.steer, 4),
        "throttle": round(out.throttle, 4),
        "dopamine": round(out.dopamine, 4),
        "speed_ms": round(speed_ms, 3),
        "edge": edge,
        "spike_rate": round(out.spike_rate, 6),
        "regions": {k: round(v, 5) for k, v in out.regions.items()},
        "nearby_agents": nearby_agents or [],
        "fov_deg": fov_deg,
        "fov_radius_m": fov_radius_m,
        # -- Phase 3 additions, all three watchable live ------------------
        # Signed degrees from the fly's nose to the route target: the number
        # that says whether the compass is right, next to `steer`, which
        # says whether the brain is acting on it.
        "heading_error_deg": round(heading_error_deg, 1),
        # Positive when the remaining route distance is growing -- what the
        # wrong-way punishment reads.
        "wrong_way_ms": round(wrong_way_ms, 2),
        # What the road and its traffic allow right now, as opposed to the
        # vehicle's absolute envelope.
        "speed_cap_ms": round(speed_cap_ms, 2),
        # What the vehicle actually steers on: `steer` above is one frame of
        # a one-neuron-per-side readout, this is that readout low-passed.
        "steer_filtered": round(steer_filtered, 4),
    }


# --------------------------------------------------------------------------
# entry point -- reproduces the calibration table config.yaml cites
# --------------------------------------------------------------------------

def main() -> None:
    import json

    from dipteran.config import load_config
    from dipteran.connectome import build_adjacency, resolve_clusters

    cfg = load_config()
    built = build_adjacency(cfg)
    clusters = resolve_clusters(cfg, neurons=built["neurons"])
    mono = monoamine_rows(cfg, built["neurons"])
    print(f"[snn] {len(mono):,} monoamine rows dropped from the fast path "
          f"(they act through plasticity instead)", flush=True)

    result = calibrate_gain(built["W"], built["neurons"], clusters,
                            monoamine=mono)
    print(f"\n{'gain':>8} {'population':>11} {'network':>9} {'motor':>8}   flags")
    for m in result["measurements"]:
        flags = ",".join(k for k in ("silent", "motor_silent", "saturated")
                         if m[k])
        print(f"{m['gain']:>8} {m['population_hz']:>11} {m['network_hz']:>9} "
              f"{m['motor_hz']:>8}   {flags}")
    print(f"\nrecommended gain {result['recommended_gain']} "
          f"(network {result['recommended_network_hz']} Hz, "
          f"motor {result['recommended_motor_hz']} Hz); "
          f"config.yaml has {cfg.snn.gain}")

    out = cfg.paths.cache_dir / "lif_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[snn] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
