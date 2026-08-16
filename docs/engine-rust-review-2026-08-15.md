# Rust Engine Review Notes — 2026-08-15

> Scope: static review of `src/engine` against `docs/architecture.md` and the current codebase map in `docs/engine-codebase-map.md`.
> Verification: `cargo test` under `src/engine` passes: 7 unit tests + 49 integration tests + doc tests. The issues below are therefore review findings / missing-coverage risks, not currently failing tests.

## Executive Summary

The engine is structurally close to the intended architecture: Rust owns state transitions, legal move generation, resource-source selection, graph/resource queries, heuristic/search, PyO3 bridging, and NN encoding. The major architecture drift is documentation: `docs/architecture.md` still describes an older 715-slot scalar policy/value design, while the code has moved to a 1316-slot branched policy and 4-player value head.

The highest-risk correctness gap is in the boundary between legal move generation and move execution. `legal_moves` generally emits executable moves, but several `execute_*` functions do not fully re-check legality, card index validity, connection/slot bounds, affordability, or transactional rollback. Because PyO3 exposes `GameState.apply_move(canonical)` and `apply_move_raw(canonical)` directly (`src/engine/src/pymod.rs:163`, `src/engine/src/pymod.rs:180`), malformed or stale canonical moves can mutate the state illegally, return success without discarding a card, return an error after partial mutation, or panic.

## P0 Correctness Issues

### 1. Raw execution can perform actions without discarding a valid card

`discard_card` silently returns when `card_index >= hand.len()` (`src/engine/src/rules.rs:347`). Most action executors call it after mutating state and do not prevalidate the card index:

- `execute_network` spends/builds link before `discard_card` (`src/engine/src/rules.rs:936`, `src/engine/src/rules.rs:953`, `src/engine/src/rules.rs:959`, `src/engine/src/rules.rs:961`).
- `execute_network_double` commits links/resources, spends money, then discards (`src/engine/src/rules.rs:1393`, `src/engine/src/rules.rs:1395`, `src/engine/src/rules.rs:1398`).
- `execute_develop` consumes iron and removes tiles before discard (`src/engine/src/rules.rs:1472`, `src/engine/src/rules.rs:1483`, `src/engine/src/rules.rs:1494`).
- `execute_loan` grants cash / drops income before discard (`src/engine/src/rules.rs:1863`, `src/engine/src/rules.rs:1865`).
- `execute_pass` always returns `Ok` even if no card was removed (`src/engine/src/rules.rs:1925`).

This violates the rule that every action must discard the required hand cards, and it creates a real API risk because `move_codec::decode` only parses field shape, not legality. Fix direction: replace silent `discard_card` with a fallible helper, or pre-check valid card indices at the top of every executor before any mutation. Add regression tests that apply malformed canonical strings through `apply_move` and assert no state mutation.

### 2. `execute_build` can overwrite illegal slots or spend for a no-op placement

`check_build_target` has the correct build legality checks for generated moves, including slot occupancy / overbuild rules (`src/engine/src/rules.rs:553`). `execute_build` does not repeat those checks. It validates era, canal one-per-city, card, resources, and money, but does not verify:

- the requested `slot_index` exists and allows `ind`;
- an existing tile is empty or legally overbuildable;
- farm locations use the expected farm slot semantics.

The actual placement uses `GameState::place_tile`, whose docstring says it overwrites any existing tile (`src/engine/src/state.rs:639`). If a city `slot_index` is out of bounds, `place_tile` does nothing, but `execute_build` has already consumed the player tile, spent money/resources, and later discards (`src/engine/src/rules.rs:767`, `src/engine/src/rules.rs:772`, `src/engine/src/rules.rs:804`, `src/engine/src/rules.rs:839`).

Fix direction: factor build target validation into a shared `validate_build_target_for_execute` used by both generator and executor, or have executor call an equivalent target check with the actual existing tile before consuming anything.

### 3. `execute_network` lacks defense-in-depth legality and affordability checks

`get_valid_network_targets` rejects wrong-era connections, no remaining links, non-adjacent links, and unaffordable links (`src/engine/src/rules.rs:853`). `execute_network` does not re-check those constraints:

- no `conn.canal` / `conn.rail` check for the current era;
- no adjacency/network check;
- no remaining link count check before decrement;
- no affordability check before `spend_money`;
- no card validity check.

This means a raw canonical `Network` can build a wrong-era or non-adjacent link, overspend into negative money, or underflow link counts. The generator path is mostly safe; the public execution path is not.

Fix direction: enforce the same conditions from `get_valid_network_targets` in `execute_network`, but return precise errors instead of relying on membership in a generated Vec.

### 4. Multi-sell can partially mutate state and then return `Err`

`execute_sell` processes entries sequentially. If an early tile succeeds and a later tile fails, the function can return `Err` after consuming merchant beer, applying merchant bonus, consuming brewery beer, and flipping tiles (`src/engine/src/rules.rs:1677`, `src/engine/src/rules.rs:1694`, `src/engine/src/rules.rs:1716`, `src/engine/src/rules.rs:1736`). Generated multi-sell plans are dry-run validated, but raw/stale canonical moves can leave a partially mutated state despite returning an error.

Fix direction: either prevalidate the entire sell plan before mutation or implement a small transaction/undo log similar to `RailTx`. This is especially important because `apply_move_raw` reports `(summary, false)` but leaves the Python-side state object mutated on error.

### 5. `execute_develop` can consume iron before failing tile validation

`execute_develop` validates and consumes iron before checking that `ind1` and `ind2` are actually developable/available (`src/engine/src/rules.rs:1458`, `src/engine/src/rules.rs:1472`, `src/engine/src/rules.rs:1483`). If a malformed canonical chooses an unavailable second industry, the function can spend resources/money and then return `Err`.

Fix direction: validate all tile availability, same-industry remaining count, and card index before consuming iron. Then mutate.

## P1 Rule / API Issues

### 6. Scout accepts duplicate or out-of-range card indices

`can_scout` checks only hand size / wild availability (`src/engine/src/rules.rs:1872`). `execute_scout` sorts indices and removes each only if `idx < hand.len()` (`src/engine/src/rules.rs:1899`). A raw move like duplicate indices or `[0, 99, 100]` can discard fewer than 3 intended cards and still grant both wilds (`src/engine/src/rules.rs:1916`).

Fix direction: require exactly three distinct in-range indices before mutation.

### 7. Some malformed canonical moves can panic instead of returning `Err`

Examples:

- `execute_network_double` indexes `state.links[conn1]` / `state.links[conn2]` before checking connection bounds (`src/engine/src/rules.rs:1334`), then unwraps connection lookup (`src/engine/src/rules.rs:1337`).
- `execute_sell` indexes `state.city_tiles[key]` directly (`src/engine/src/rules.rs:1645`).

`move_codec::decode` accepts arbitrary `usize` values for these fields, so this is observable through PyO3. Fix direction: use `.get()` / checked connection lookup before indexing, and return `Err`.

### 8. `engine::step` advances the turn even when move application fails

`engine::step` calls `apply_move`, then always calls `advance_turn` (`src/engine/src/engine.rs:281`). Current game loops avoid this by applying moves manually and only advancing on success, but the function is public and documented as a full step. If reused, illegal moves would still consume turn flow.

Fix direction: make `step` advance only on `Ok`, or remove/rename it if the shared `game_loop::play` path is the real driver.

## P2 Architecture / Documentation Drift

### 9. `docs/architecture.md` describes stale policy/value dimensions

The architecture doc still says the fixed action table has 715 slots and double rails are 102 shared-endpoint pairs (`docs/architecture.md:108`). Current `policy.rs` documents and implements 1316 slots with all unordered rail pairs as the double-rail superset. The same architecture section describes a scalar `policy 715 + value` network (`docs/architecture.md:126`), while the current AGENTS/project state says the network returns `(type, goal, value)` and value is 4-player.

Fix direction: update `docs/architecture.md` or mark the old Phase 3 block as historical. Otherwise future work may resize networks or masks incorrectly.

### 10. `docs/architecture.md` still says graph uses BFS per query

The architecture doc says `graph` is “static adjacency + u32 bitmask BFS, zero heap allocation” (`docs/architecture.md:40`). Current code intentionally moved resource/connectivity queries to `GameState` caches, with `graph.rs` reading cached masks and free-source lists. The codebase map is current; architecture is stale.

Fix direction: align architecture with the cache-based design, including the invariant that cache maintenance is part of `state` mutation correctness.

### 11. PyO3 state is not actually serializable yet

The architecture requires state serialization for determinization, save/load, and debugging (`docs/architecture.md:68`). `GameState` is cloneable and encodable to tensors, but there is no serde/json state snapshot API in `pymod.rs`. This is not an immediate rules bug, but it matters for TTS ingestion, replay interchange, and reproducible bug reports.

Fix direction: define a stable state snapshot schema separately from move canonical strings; include deck/discard/played/wild piles if the snapshot is meant to support determinization.

## Suggested Next Tests

Add a focused `raw_execution_rejects_invalid_moves_without_mutation` test group:

- invalid `card_index` for `Loan`, `Pass`, `Network`, `Develop`, `Sell`, `NetworkDouble`;
- `Build` into an incompatible occupied slot and out-of-range slot;
- `Network` on non-adjacent / wrong-era / unaffordable connections;
- duplicate and out-of-range `Scout` indices;
- malformed `Sell` with one valid first tile and invalid second tile, asserting rollback/no partial state change;
- invalid `NetDouble` connection id and invalid `Sell` key, asserting `Err` rather than panic.

Use whole-state snapshots or targeted invariants (`money`, `hand.len`, `city_tiles`, `links`, resource markets, merchant beer, pending bonus, discard pile) to prove failed raw moves are atomic.
