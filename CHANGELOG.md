# Changelog

## 0.2.0 — ENGRAM v2 (trace model)

A redesign from first principles of human memory; not backward compatible.

- **One table, one formula.** A memory is a trace with `lastRecall` and
  `stability`; `R = 2^(−Δt/S)`, retrieval grows stability by
  `1 + gain·a·(1−R)` (spacing effect × cue activation). Tiers, mass, refractory
  timers, generation counters, conflict rings, dream logs, tombstones and the
  per-turn `maintain()` are gone.
- **Nothing is overwritten at write time.** Paraphrases, updates and
  contradictions become labile pairs (reconsolidation: the old trace is
  reactivated too) that the dream adjudicates with both timestamps in view.
- **Forgetting = eviction of the lowest strength (`S·R`)** past a 3-day grace
  period (hippocampal buffer); floods evict their own members.
- **Injection is exposure, not use**: half-activation strengthening on recall,
  completed by the new `cite(reply)` for the memories the LLM quoted.
- **Dream** seeds clusters from labile traces (most stable first), ≤ 8 members
  per verdict, `keep` / `replace`, gists inherit the strongest member plus live
  evidence with strength conservation, confabulation guard by cosine. A settled
  store makes no LLM calls.
- **Salience** (emotional weight, 0–10) scales initial stability; `cosineFloor`
  calibrates cue activation to the embedding model.
- `MemoryStore` shrinks to six operations; the store is loaded into RAM at
  start and stale-model vectors are re-embedded from text (failure-tolerant).
- Ids carry a 50-bit millisecond timestamp (valid past year 37,000); local-time
  formatting handles negative offsets and any year.
- `package:crypto` dependency removed — the package has no dependencies.
- Test suite rewritten, including a 3000-virtual-year simulation and a
  cross-language conformance scenario shared with the Python reference.

## 0.1.0

Initial release: pure-Dart port of the ENGRAM v1.1 engine.
