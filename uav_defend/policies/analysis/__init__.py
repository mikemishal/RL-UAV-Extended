"""ANALYSIS / ORACLE controller package (Phase 6 of the learning-benefit
diagnostic study). Nothing in this package is a deployable policy: every
controller here either requires ground-truth state via an explicit,
non-standard method signature (never through `act(obs, info)`), or is
otherwise clearly documented as a diagnostic-only tool. Do NOT register
anything here in `uav_defend.policies.registry`.
"""
