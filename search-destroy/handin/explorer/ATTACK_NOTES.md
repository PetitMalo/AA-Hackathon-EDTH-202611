# Terminal attack mode (experimental, off by default)

`ExplorerConfig.use_attack = False` in `config.py`. The default policy is the **tuned MPC**
(`approach_speed_m_s = 12`, `w_term_vel_terminal = 0.1`, `a_max_xy_m_s2 = 12`, `max_tilt_deg = 50`,
glide slope `0.15`): it reaches the box instead of creeping, no longer stops at the horizon, and
succeeds on the tested starts.

## What `use_attack = True` adds

When the target is tracked, the drone is below `engage_max_alt_m`, the sighting is fresh
(`< lost_timeout_s`), and the straight segment drone→box clears known-free voxels (ignoring the
ground layer), `_attack_action` bypasses the corridor/MPC and pitches near-max at the box.

## Known issue (why it is off)

On `field@v1` seed 0 the attack fires at ~10 m, z ≈ 5 m. The dive authority (`a_max_z_m_s2 = 6`)
is too weak relative to the 12 m/s² horizontal sprint, so the drone skims over the box (arrives at
z ≈ 2.5 m, box top = 1.55 m) and collides just past it. Aiming at the box top (`box_top_clamp_m = 1.5`)
instead of the centre did not fix it — the dive is still too slow.

## Next try (for whoever picks this up)

- Make the horizontal accel **proportional to the dive authority**, not constant: cap `accel_h` so
  `time_to_close ≈ time_to_drop`. There is a stub of this in `_attack_action` (`v_h_need`, `accel_h`)
  but it is not tuned.
- Or: enter the attack only once the glide slope has brought the drone within ~1 m of the box top
  (`engage_max_alt_m ≈ 2.5`, `engage_dist_m ≈ 6`), so the dive is short and the sprint has no room
  to overshoot.
- Or: in the settle zone, drop throttle below hover and let gravity finish the last meter —
  currently `accel_to_sticks` maps a dive to extra collective thrust, which fights the descent.

## Tunables (all in `AttackConfig`, `config.py`)

`engage_dist_m`, `engage_max_alt_m`, `lost_timeout_s`, `accel_m_s2`, `tilt_deg`,
`box_top_clamp_m`, `aim_above_box_m`, `ground_layer_m`, `settle_dist_m`, `clear_voxels_ahead`.

## Files changed on this branch (`fast-terminal-run`)

- `handin/explorer/config.py` — `ControlConfig`, `MPCConfig` retuned; new `AttackConfig`,
  `phase_of_mode`; `use_attack` flag.
- `handin/explorer/mpc.py` — second pre-factored cost (`P_terminal`, `K_inv_terminal`) selected by
  `set_terminal()`; cost read from the attribute each ADMM iter so mid-loop rho refactors stay
  consistent.
- `handin/explorer/corridor.py` — `CorridorPlanner.step(..., terminal=)` forwards the flag.
- `handin/explorer/controller.py` — `accel_to_sticks(..., tilt_deg=)` override for the attack.
- `handin/explorer/explorer.py` — terminal flag plumbed; `_attack_action`, `_path_clear_to`,
  `_publish_phase`; reports `phase = "exploration"|"terminal"` so the evaluator's mission split works.
- `handin/view_explorer.py` — `attack` mode colour.