# notes.md — scratch decisions

Public scratchwork. Will become the basis of the README. Informal on purpose.

## what this is

RL env for an LLM-based travel agent. Three-hour build. Synthetic world. Tool-call action space. Rule-based reward + LLM persona.

one-liner:
A deterministic synthetic travel world (`world.py`) with a structured persona (`persona.py`) that's voiced by an LLM, wrapped in a Gym-style state machine (`env.py`) that exposes a ~10-tool action space, scored by composable rule-based rewards (`reward.py`), and driven by either an LLM rollout or a heuristic baseline.

the env owns the truth, the LLMs handle natural language and back-and-forth interaction to simulate an actual conversation. but persona's preferences are structured data, reward is actually computed from state. LLM-judge is just a final diagnostic, not the primary signal

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
3. `budget_efficiency` — asymmetric: linear ramp up to the midpoint, quadratic decay above it down to 0 at the hard cap (so cheapness pays a continuous price, aggressiveness only loses near the cap)
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
- **`add_to_itinerary` enriches `slot.meta` with item attributes** (stars, neighborhood, amenities for hotels; stops, cabin, overnight for flights; category for activities). Otherwise reward.py would have to call `world.get_details(item_id)` inside its scoring loop — coupling reward to world for no good reason. Slot is self-describing for downstream consumers.

---

## reward.py decisions (during build)

This was the README's flagged "hardest part." Spent extra cycles on the exploit audit.

### shape

- **Sparse, terminal-only.** Reward returned by `step` is 0 every turn except the last; full `RewardBreakdown` is in `info["reward_breakdown"]`. "Good third turn" isn't a thing in travel planning.
- **Multiplicative gate × additive soft sum**. Failing any hard constraint zeros everything except the step penalty. Soft components add inside the gated bracket.
- **Step penalty is post-gate** (not multiplied through). So even an empty/failed episode is dinged for dithering. Free for the first 5 turns.

### components

- **hard_constraint_gate {0, 1}**: budget ≤ cap; ≥1 booked hotel; ≥1 outbound + ≥1 return flight on the right routes; `no_overnight_flights` respected; `max_stops` respected; `required_amenities` present on all booked hotels. Multiplicative.
- **preference_coverage [0, 1]**: weighted score over persona soft_prefs. Axes scored: `act_<category>` (linear up to 2 in-category activities — two beats one, two beats four; anti-greedy on the easy pref), `hotel_stars` (distance from target, tolerance-aware), `flight_cabin` (rank distance, tolerance-aware), `flight_max_stops` (1.0 if within target, linear decay above). Persona-omitted axes contribute nothing.
- **budget_efficiency [0, 1]**: asymmetric bell. **Tolerances are persona-supplied** — `budget_low_ratio` and `budget_high_ratio` from `profile.tolerances`. Linear ramp from 0 (spent nothing) up to 1.0 at the midpoint, quadratic decay above the midpoint down to 0 at the hard cap (ratio=1.0). Default 0.55–0.95 → midpoint 0.75. The asymmetry is what makes "always cheapest" lose: undershooting now pays a continuous price (ratio=0.3 scores ~0.4, not flat-zero), while spending up toward the persona's natural ratio is the gradient direction.
- **coherence [0, 1] (partial)**: 5 checks — outbound route, return route, hotels in dest, activities in dest, dates aligned. Each check is 0 or 1; score = sum/5. Partial credit (not 0/1 binary) so a mostly-coherent itinerary doesn't get crushed by a single date typo.
- **recovery_quality [0, 1] or None**: weighted 0.4 price_efficiency + 0.4 time_efficiency + 0.2 downstream_preservation. **None if no disruption fired** (remaining weights renormalized). Brutal: never rebooking = 0.
- **step_penalty ≥ 0**: `0.005 × max(0, steps - 5)`. So 10 turns = -0.025, 20 turns = -0.075, 40 turns = -0.175. Meaningful at the high end, near-invisible for snappy episodes.

### dynamic weight renormalization

Two regimes:
- Disruption fired → composition uses all four weights as-is (0.35 + 0.20 + 0.20 + 0.25 = 1.00).
- No disruption → recovery_quality is None; remaining three weights renormalized to sum to 1 (preference 0.467, budget 0.267, coherence 0.267).

Why renormalize: keeps reward magnitudes comparable across episode types. Otherwise no-disruption episodes would systematically score 25% lower for "lucking out" — perverse.

### the exploit audit (verified with the smoke test)

| Exploit | Defense | Verified? |
|---|---|---|
| Always cheapest flight (multi-stop overnight) | hard.no_overnight + hard.max_stops gates kill it for luxury/business/family; for others, preference_coverage on cabin/stops penalizes it | ✅ luxury seed=0: gate=0, total=-0.025 |
| Highest-rated hotel | book tool rejects if over cap (env-side); if it fits, budget_efficiency decays quadratically above the midpoint and hits 0 at the hard cap | ✅ seed=42: gate=0, budget=0, total=-0.025 |
| Finish in one step (submit immediately) | gate=0 with empty itinerary; reward=0 - step penalty | ✅ total=0 with steps=3 |
| Ignore disruptions | recovery=0 + coherence fails (now missing a flight leg) + gate fails (incomplete) | structurally — triple penalty by construction |
| Spam search | world.py search idempotence + step_penalty growth | structural |
| Book and unbook in a loop | tentative is free, unbooking refunds — but final state is what counts; turns burn step_penalty | structural |
| Submit empty | gate=0, reward=0 minus step_penalty | ✅ verified |

### what we deliberately did NOT do

- **Did not blend the LLM judge into the training reward.** Logged separately in eval. The headline diagnostic is `corr(rule_reward, judge_score)`. Low corr → blind spot. High corr → good proxy. This is the more interesting result either way.
- **Did not normalize the persona's soft_prefs weights to sum to 1.** They're already normalized in persona.py for activity weights; mixing them with hotel_stars / flight_cabin scoring (different "weight" semantics) gets messy. Each contribution carries its own scaling.
- **Did not make coherence binary 0/1**, despite the plan saying so. Partial credit (fraction of 5 checks passing) is fairer to "mostly right" itineraries and produces a smoother training signal.

### budget bell math

```
ratio = spent / cap
midpoint = (low_ratio + high_ratio) / 2
if ratio <= midpoint:
    score = ratio / midpoint                                 # linear ramp
else:
    score = max(0, 1 - ((ratio - midpoint) / (1 - midpoint))²)  # quadratic decay
```

So at default tolerances (0.55, 0.95): midpoint 0.75, score = 1.0 at ratio=0.75, score = 0.4 at ratio=0.3 (linear), score = 0.84 at ratio=0.85, score = 0.0 at ratio≥1.0. Linear ramp so undershooting pays a continuous price; quadratic decay so penalties are gentle near the peak and steep near the hard cap.

The shape used to be a symmetric quadratic bell that hit 0 at both `low_ratio` and `high_ratio`. That made "spend almost nothing" and "spend near the cap" structurally identical from the agent's perspective — combined with the hard gate's cliff at ratio=1.0, undershooting was strictly safer than aggressive spending, which is most of why "always pick the cheapest" was outscoring smarter policies. The asymmetry flips that.

### things worth a /reward-shaping analysis later

- Vary `(preference, budget, coherence, recovery)` weights on a grid; plot how baseline rewards shift. README enhancement #1.
- Sensitivity of total reward to `budget_low/high_ratio` (persona archetype tightening).
- Compare `corr(rule, judge)` across personas — is the rule-based proxy archetype-fair?

---

## baselines.py decisions (during build)

- **Stateless functions, not classes.** Match the skeleton signatures and avoid persistence machinery. State is derivable from the obs each turn — what's booked, what's tentative, what was last_result. Keeps the API trivial for eval.py to call.
- **`profile` is an optional kwarg.** When eval.py uses signature inspection it only passes `rng`, so `profile=None` is the default path. Heuristic still works (uses obs.state.clock for trip params and reasonable defaults for picks).
- **All three policies share one state machine** (`_flow_action`) that builds the trip in a deterministic order: disruption recovery → outbound → return → hotel → book base → activities (heuristic only) → propose (heuristic only) → submit. Mode flag (`cheapest` vs `heuristic`) toggles selection behavior + activity/propose phases. No code duplication across baselines.
- **Random is truly random over context-valid actions.** Searches with sensible defaults, picks add/book/swap/remove/propose/submit by uniform sampling from valid candidates. Often submits with an empty itinerary (gate=0). That IS the floor we want to surface.
- **Cheapest deliberately skips activities and propose_to_client.** The whole point of cheapest as a baseline is "minimize spend at all costs." Adding activities or iterating with the client both cost money or turns. We let it run its degenerate strategy and observe what the reward function does.
- **Heuristic without profile falls back to "cheapest among hard-feasible."** Without persona soft prefs there's nothing better to do. The differentiator is that heuristic *still adds activities* and *still proposes once* — that's where it pulls ahead of cheapest on pref_coverage.
- **`_pick` filters by hard_constraints first**, then runs mode-specific selection on the survivors. If nothing survives, it falls back to the unfiltered set so we always return a valid id (gate will fail, but we don't crash).

### supporting change in env.py

Added `origin`, `dest`, `group_size`, `no_overnight_flights`, `max_stops`, `required_amenities` to `obs.state.clock`. Rationale: these are *publicly stated* by the client in the request text. Exposing them structurally lets non-LLM policies filter cleanly without parsing free text. **Soft prefs stay secret** (those are inferred from tone / preference statements; the agent has to actually read the request to score well on them).

### IATA ↔ city name matching

Both reward.py and baselines.py need to match flights tagged with IATA codes (`SIN`) against personas carrying city names (`Singapore`). Each module ships its own tiny `_IATA_TO_NAME` map derived from `world.CITIES`. Small DRY violation but the alternative (a shared helper module) felt heavier for a 4-line lookup.

### empirical baseline ranking (20 seeds, scripted persona)

```
              reward    gate%   pref   budget   coh
random       -0.014       0%   0.03    0.05    0.03
cheapest     +0.402      75%   0.40    0.27    0.79
heuristic    +0.428      80%   0.50    0.29    0.80
```

Strict ordering on every metric: heuristic > cheapest > random. The reward function differentiates as designed.

The gap between cheapest and heuristic is mostly preference_coverage (0.40 → 0.50), driven by heuristic adding two activities per trip. If eval.py passed `profile` to baselines (currently it doesn't — clean separation), heuristic would pull further ahead via soft_pref-aware ranking. Worth flagging as a small enhancement.

### a surprising finding from the 100-seed eval pass

Original numbers (symmetric budget bell, pre-asymmetric refactor):

```
baseline     reward  gate%   pref  budget   coh   disrupt   recovery
random       -0.013     0%   0.01   0.01   0.02    0/100   - (n=0)
cheapest     +0.405    78%   0.42   0.20   0.85    1/100   0.00 (n=1)
heuristic    +0.412    79%   0.51   0.21   0.85   26/100   0.60 (n=26)
```

**Cheapest dodges disruptions almost entirely.** Only 1/100 episodes triggered a disruption that actually fired during the episode — because cheapest finishes in ~8 turns, and disruptions are scheduled `fires_at_turn = current_turn + rng.integers(1, 5)`. The episode ends before the trap springs.

Heuristic takes more turns (search activities + add + book + propose), giving disruptions time to manifest — 26/100 episodes. Recovery quality of 0.60 over those 26 means it's rebooking but not always optimally (some turn-delay, some price overshoot).

This isn't an exploit — it's a real dynamic: speed reduces disruption exposure. A real travel agent has the same tradeoff (book and bail vs. iterate with the client). The reward structure tolerates this because:
- Cheapest still loses on `preference_coverage` (no activities → 0.42 vs heuristic's 0.51).
- IF a disruption did fire and cheapest ignored it (booked the cancelled flight, never rebooked), the triple penalty (recovery=0, coherence fails on missing leg, gate fails on incomplete itinerary) still kicks in. The "ignore disruption" exploit is structurally defended; the "finish before disruption fires" path is just a feature of the dynamics.

Worth flagging in the README: if we wanted to force cheapest to confront disruptions, we'd schedule them with `fires_at_turn = current_turn + 0..2` (same-turn possible) or raise the base rate. Trade-off: more disruptions also makes episodes longer and increases LLM rollout cost. Current calibration favors a clean signal.

### test suite

`uv run pytest tests/` — 88 tests, 0.31s, no network. Covers:
- world: determinism, idempotence, filters, disruption distribution
- persona: profile sampling, scripted voice, mismatch detection
- env: 5-tuple step return, tool dispatch, budget/itinerary transitions, truncation
- reward: every component + every documented exploit defense
- baselines: each runs to completion, aggregate ordering holds

### archetype-aware budget tolerance bands

Original `budget_low_ratio=0.55, high=0.95` was the same for every archetype. That under-rewards budget personas (their actual spend is well below 0.55 of cap) and would over-reward luxury ones if cheap. Added `ARCHETYPE_BUDGET_BANDS` in persona.py:

```
budget       0.40-0.80   (budget travelers spend less, cap their "good zone" lower)
luxury       0.80-0.98   (expect to spend most of the cap)
foodie       0.55-0.92
family       0.60-0.95
history_buff 0.50-0.90
business     0.70-0.98
```

100-seed re-run:
- before: cheapest=0.405, heuristic=0.412, gap=+0.007
- after:  cheapest=0.400, heuristic=0.411, gap=+0.011

Small effect. Most of the heuristic-over-cheapest delta lives in `preference_coverage` (activities), not `budget_efficiency`. Honest takeaway: the bands are a more principled model of persona budget expectations, but tightening them isn't enough to make cheapest look obviously bad. To widen the gap meaningfully, eval would need to pass `profile` so heuristic can use soft prefs in selection — that's the bigger lever.

### asymmetric budget bell

The symmetric quadratic bell was the bigger lever after all. With it hitting 0 at both `low_ratio` and `high_ratio`, both baselines were scoring ~0.20 on `budget_efficiency` (cheapest because it sat well below low_ratio, heuristic because it was only slightly above), so the axis wasn't differentiating them at all. Worse, the symmetry meant undershooting and overshooting looked equally bad to the agent — but only overshooting also risked tripping the hard gate at ratio=1.0. Net incentive: stay safely cheap.

Replaced the shape with a linear ramp up to the midpoint, quadratic decay above it down to 0 at the hard cap. Same signature, same midpoint definition (so `ARCHETYPE_BUDGET_BANDS` keeps working unchanged). 100-seed re-run (scripted persona, seed=42):

```
                cheapest   heuristic   gap
before          0.405      0.412       +0.007
after           0.475      0.542       +0.067   (≈10× wider)
```

Both baselines' absolute rewards went up — the ramp now gives partial credit for any spending, where the old bell flat-zeroed everything below low_ratio. The headline is the gap: heuristic spends more, so it sits closer to the peak and pulls ahead on the budget axis (0.556 vs 0.461) where before both were ~0.20. The gap on budget alone (0.095) is bigger than the entire cheapest-vs-heuristic reward gap was under the old bell.

## eval

`python eval.py --episodes 50 --seed 42 --baselines random,cheapest,heuristic [--llm-judge] [--persona-mode llm|scripted]`

Output: per-baseline metrics table + reward decomposition. Expected ordering across baselines:
- random worst on `preference_coverage`
- cheapest worst on `preference_coverage` *and* `recovery_quality` (no reason to rebook well)
- heuristic best across the board
- all three show non-trivial `coherence` failures occasionally — env actually checks geometry, doesn't just instruct against it

`--render` prints each turn's action + obs snippet for spot-checks. Weights logged with every run. If `--persona-mode` is run both ways, eval breaks out LLM vs scripted to surface persona-induced variance.

## optional enhancements landed

From the README's prioritized enhancement list:
- ✅ **#2 disruption engine** (simple, not cascading) — probabilistic flight cancellation, rebook flow, recovery_quality scoring
- ✅ **#3 client persona system** (full) — 6 archetypes × varied prefs/budgets/flexibility/comm-styles, sampled deterministically per seed
- ⚠️ **#1 reward-shaping** (partial) — configurable `RewardWeights`, every exploit defense verified empirically; no formal weight-sweep grid (left as future work)
- ✅ **#6 environment visualization** — `viz.py` runs one baseline against the env and renders a live 4-panel terminal dashboard via `rich.Live`: header (seed/archetype/turn/trip), budget progress bar, color-coded itinerary table, last-action panel. At end-of-episode the action panel is replaced with horizontal reward-component bars. Usage: `uv run python viz.py --seed 42 --baseline heuristic`.

Not shipped: curriculum (#4) and config-driven YAML (#5).

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
