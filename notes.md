# notes.md — scratch decisions

Public scratchwork. Will become the basis of the README. Informal on purpose.

## what this is

RL env for an LLM-based travel agent. Three-hour build. Synthetic world. Tool-call action space. Rule-based reward + LLM persona.

---

## interface

- Core env = Gymnasium-style. `reset` / `step` / `close`. **No gymnasium dep** — `gym.spaces.Box/Discrete/Dict` don't fit variable-length text and structured inventory.
- LLM rollout helper sits *on top* of the env. Verifiers-style. ~50 lines. Demonstrates we understand both abstractions and what each is for.
- Composes: env = the simulator; rollout = the training-loop adapter.

## action space

- JSON tool calls: `{"tool": "search_flights", "args": {...}}`
- 11 tools. Search → details → tentatively add → propose → react to feedback → swap → book → submit.
- The set: `search_flights`, `search_hotels`, `search_activities`, `get_details`, `add_to_itinerary`, `remove_from_itinerary`, `swap`, `book`, `propose_to_client`, `message_client`, `submit_final`.
- Tentative vs booked is a real distinction. `book` debits budget, locks inventory, makes flight disruption-eligible. Gives the agent room to plan before committing.

## observation

Three layers:
1. **Static** (constant per episode): client request text + tool catalogue. Goes in system prompt.
2. **State snapshot** (refreshed every turn): itinerary, budget, clock, pending events.
3. **Last action result**: search results, booking confirms, feedback, disruption events.

Plus history (with prior search results pruned to summaries — controls token growth without losing the trajectory).

## persona

- **Profile** = structured, deterministic from seed. Hard constraints (budget, dates, group) + soft prefs (weighted vector) + tolerances + communication style. **Reward reads this. Agent never sees it.**
- **Voice** = LLM (OpenRouter). Rewrites profile into a free-text request. Voices feedback in character. Cached on disk for reproducibility.
- Cleanly separates "what the persona wants" (machine-readable) from "how the persona talks" (LLM realism).
- Mismatch detection is deterministic. The LLM only voices complaints we've already structurally identified.

## reward

5 components, all bounded:
1. `hard_constraint_gate` — multiplicative 0/1 (budget honored, dates match, slots filled)
2. `preference_coverage` — weighted dot product against persona soft prefs
3. `budget_efficiency` — concave, peaks at ~0.85 of cap (penalizes both "way under" and "right at cap")
4. `coherence` — 0/1 (geometric + temporal feasibility)
5. `recovery_quality` — conditional on a disruption firing
6. small step penalty (anti-dither)

Total = `gate × (Σ w·components) - step_penalty`.

### exploit defenses (by design, not by instruction)

- Always-cheapest flight → persona has hard prefs (no red-eyes) that gate the constraint
- Always-highest-rated hotel → hard budget cap + budget_efficiency penalizes overshoot
- Finish in one step → reward only at `submit_final`, which requires coherence pass
- Ignore disruptions → recovery is a real fraction of reward AND failing to rebook fails coherence

### LLM judge

Optional. Runs at end-of-episode. **Logged separately, never blended into training reward.**

- Blending makes training expensive (LLM call per rollout) and partially trusts a stochastic signal we can't audit.
- Used instead as a *diagnostic*: report `corr(rule_reward, judge_score)` in eval. Tells us whether our proxy reward is actually correlated with what humans care about. Low corr → blind spot. High corr → good proxy. Either result is useful.

## world

No real APIs. Reasons:
- Reproducibility (RL needs seeded determinism)
- Speed (no network in inner loop)
- Control (we want to engineer the distribution to *test specific competencies*)
- README explicitly says no starter data

Key insight from the README: **"a flat data distribution produces a flat agent."** Engineering the distribution shape is the whole game.

### what good world data should do

Properties we're baking in (and why):

1. **No dominated options in search results.** Every flight in top-k is the cheapest at its quality tier, or highest-quality at its price tier. Forces real tradeoff reasoning. (Implemented via anti-correlated price/quality noise.)
2. **Persona-aligned and -misaligned options coexist.** Foodie in Tokyo sees food experiences AND history AND nightlife in `search_activities`. Forces filtering and ranking by preference.
3. **Geography with consequences.** Neighborhoods have transit times. Cheap hotel might be 90 min from airport with a 6am flight. Agent that ignores geography fails coherence.
4. **Uneven availability.** Some routes have many flights, some few. Some days the perfect option doesn't exist. Forces scarcity handling.
5. **Search idempotence.** Same `search_flights(args)` returns same results within an episode. Prevents spam-search-to-get-lucky exploits. Forces query variation.
6. **Disruption risk asymmetry.** Early-morning flights cancel more (real pattern). Connections cancel more than direct. Gives agent a reason to consider risk *during initial booking*.

### data shape

- ~15 real cities (Tokyo, Paris, NYC, etc.) with IATA codes, coords, 3–5 famous neighborhoods, cost multipliers.
- Flights procedurally generated. Price ~ `f(distance, time_of_day, advance, stops, cabin) + anti-correlated noise`.
- Hotels per-city. Price ~ `city_mult × (40 + 35·stars²) × neighborhood_premium + noise`.
- Activities: small per-city library of real named anchors ("Senso-ji Temple", "Louvre") + procedural fill.
- Cost constants anchored to real ballparks (Tokyo 4-star ≈ $250/night) so a $40k trip looks absurd and $1k looks tight.

## disruption

- Probabilistic flight cancellation after booking. `p ~ 0.15` per booked flight, tuned so we get ~1 disruption per episode in expectation.
- Schedules a `pending_event` at a randomized future turn.
- Agent must call `search_flights` + `book` to rebook.
- Recovery quality scored on: turns-to-rebook, price delta, downstream preservation.

## scope cuts

- Single-city return trips only. Multi-city = stretch goal.
- One disruption per episode. Cascade chains = stretch.
- Group dynamics (multiple travelers with conflicting prefs) = stretch.
- Real-time visualization = no.
- Config YAML = no — sensible defaults + a few CLI flags.

## baselines

- **Random valid action** — floor.
- **Cheapest-feasible** — the "cheap exploit" probe. Reward function should structurally push back; if it doesn't, our reward is broken.
- **Persona-aware heuristic** — ceiling for non-learned policies. Respects soft prefs.
- (Optional) LLM zero-shot if time.

---

## env.py decisions (during build)

- **Modern Gymnasium 5-tuple API**: `obs, info = reset(seed=...)` and `obs, reward, terminated, truncated, info = step(action)`. Post-v0.26 convention. The rollout helper can collapse `terminated | truncated` to `done` for the LLM; env stays canonical.
- **No `done` field in obs**. That's what terminated/truncated are for. Avoids the "two sources of truth" trap.
- **Reward is 0 every step except the terminating one.** Sparse but principled — there's no meaningful per-step reward signal for travel planning ("good third turn" isn't a thing). The full `RewardBreakdown` is exposed via `info["reward_breakdown"]` for inspection.
- **Tool dispatch via handler dict** (`self._handlers = {name: method}`), not if/elif. Cleaner, and adding/removing tools is one line in two places (registry + dict).
- **Slots are free-form strings.** Agent picks the names. No env-side validation of slot taxonomy — the reward function's coherence check does the structural work. Keeps env permissive, reward strict.
- **Two-tier validation**: env rejects *immediate* sanity violations (unknown item_id, double-book, budget overshoot on book) and returns them as tool errors. Reward rejects *structural* violations (incomplete itinerary, geographic infeasibility) by zeroing the gate. Clean separation.
- **Tentative vs booked is meaningful state.** `add_to_itinerary` is free and reversible. `book` debits budget AND triggers the disruption roll. If a flight cancels, refund the slot price; agent has to rebook to recover.
- **Hotel pricing**: stored as `price_per_night` in world.py; expanded to total at `add_to_itinerary` time using `nights = days(depart_date, return_date)` from the persona profile. Agent doesn't need to specify night counts — they're implicit in the trip dates.
- **Disruption tick happens *before* tool dispatch each step.** So when a flight cancels on turn 12, the agent sees it as `last_result.disruption_fired` even if they were doing something else this turn. Avoids the "agent must explicitly poll for events" anti-pattern.
- **Tool errors don't crash the env.** Any exception inside a handler is caught and surfaced as `{"ok": false, "error": "..."}`. Real RL needs robustness — a malformed JSON from a poorly-trained agent can't take down the whole episode.
- **History pruning lives in `_render_obs`, not in the stored history.** Internal `self._history` keeps everything (for debugging, eval analysis). The exposed obs prunes the *last 3 turns kept in full, older search results collapsed to summaries* version. Bookings/feedback/proposals always stay in full because they're cheap and load-bearing for context.
- **Auto-downgrade `persona_mode="llm"` to `"scripted"` if `OPENROUTER_API_KEY` unset.** No-key path always works — important for CI and for reviewers without credentials.
- **`_action_from_input` accepts both `dict` and `Action`.** Rollout passes dicts (parsed from LLM JSON); programmatic baselines can pass `Action` directly. No JSON parsing in the env — that's the rollout's job.
- **`info` dict shape mirrors Gymnasium expectations**: stable keys (`turn`, `reward_breakdown`, `disruption_fired_turn`). Reward breakdown is converted to a plain dict (not a dataclass) for serializability.

## eval

`python eval.py --episodes 50 --seed 42 --baselines random,cheapest,heuristic [--llm-judge] [--persona-mode llm|scripted]`

Output: per-baseline metrics table + reward decomposition. Expected ordering across baselines:
- random worst on `preference_coverage`
- cheapest worst on `preference_coverage` *and* `recovery_quality` (no reason to rebook well)
- heuristic best across the board
- all three show non-trivial `coherence` failures occasionally — env actually checks geometry, doesn't just instruct against it

`--render` prints each turn's action + obs snippet for spot-checks. Weights logged with every run. If `--persona-mode` is run both ways, eval breaks out LLM vs scripted to surface persona-induced variance.

## what we'd do with more time

- Multi-city itineraries
- Cascading disruptions (flight → missed connection → hotel no-show → rebook chain)
- Curriculum (simple → complex episodes)
- Config-driven world (non-engineers can add destinations, pricing rules, disruption types)
- Real API connectors (Amadeus, Booking.com) — would replace `world.py` search functions; persona/reward/rollout untouched (the seam already exists)
- LLM persona reactions during episode (currently only at proposal time)
- Reward weight sweep with regression analysis on behavior change

## scaling to real APIs

The seam already exists. `world.py` exposes `search_flights/hotels/activities` and `get_details`. Swap the synthetic generator for Amadeus/Booking adapters; everything above (env, persona, reward, rollout) is unchanged. Caching, rate-limiting, and async become the new concerns — solvable with standard infra, not env-design problems.
