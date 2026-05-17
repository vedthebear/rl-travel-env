# AfterQuery — AI Travel Agent RL Environment

An RL training environment for an LLM-based AI travel agent. A deterministic synthetic world (`world.py`) is populated with personas whose preferences are structured data but whose voices are LLM-generated (`persona.py`). (This mimics real-world interactions and different interaction styles, without relying on entirely LLM-generated personas). A Gymnasium-style state machine (`env.py`) exposes an 11-tool action space the agent uses to search inventory, compose an itinerary, propose, react to feedback, and handle a mid-trip disruption. (All simple python functions)! Episodes are scored by composable rule-based rewards (`reward.py`) with an optional LLM judge as a diagnostic — never as the training signal. Three baselines (`baselines.py`) plus an LLM rollout (`rollout.py`) run through an evaluation harness (`eval.py`) that reports per-component reward decompositions. A test suite (`tests/`, 88 tests, 0.3s, no network) covers the load-bearing invariants. A live terminal visualization (`viz.py`) renders one baseline's episode in real time.

---

## Quick start

### Setup (one-time)

```bash
uv sync --extra dev

# Optional: drop your OpenRouter key in .env so the LLM scripts pick it up.
echo "OPENROUTER_API_KEY=sk-or-..." > .env
```

The scripts below call `.venv/bin/python` directly and source `.env` themselves, so you don't need to remember either step at the call site.

### Live demos

Six rich-terminal dashboards, trivial → complex. Each script auto-loads `.env` and gracefully skips itself if the API key isn't set.

```bash
bash scripts/viz_01_help.sh                  # CLI sanity (instant)
bash scripts/viz_02_random_fast.sh           # panels-draw test (instant)
bash scripts/viz_03_cheapest.sh              # ~8 turns, watchable
bash scripts/viz_04_heuristic_disruption.sh  # flight cancel + rebook visualized
bash scripts/viz_05_heuristic_llm_persona.sh # LLM-voiced client request
bash scripts/viz_06_llm_agent.sh             # LLM drives the agent itself
```

### End-to-end verification

Eight graduated tests, simple → complex. `test_all.sh` runs them in order and stops on first failure.

```bash
bash scripts/test_all.sh --skip-llm   # L0-L4, no API key needed (~2s)
bash scripts/test_all.sh              # all 8 levels (adds L5-L7 LLM round-trips)
```

Individual levels (`test_00_imports.sh` … `test_07_llm_judge.sh`) are runnable standalone if you want to drill into one.

### Manual commands

For when you want more control than the scripts give:

```bash
set -a && . ./.env && set +a   # load OPENROUTER_API_KEY into the shell

# Three baselines × 50 seeds, scripted persona, no API key needed
uv run python eval.py --episodes 50 --seed 42 --baselines random,cheapest,heuristic

# LLM-voiced persona + judge correlation diagnostic
uv run python eval.py --episodes 20 --seed 42 --persona-mode llm --llm-judge

# LLM agent driving the live rich dashboard
uv run python viz.py --baseline llm --seed 42 --turn-delay 0.2

# Pytest directly (88 tests, ~0.3s, no network)
uv run pytest tests/
```

---

## How we read the task

The README that came with this take-home called reward design "the hardest part" and warned us to engineer the world so the obvious exploits *can't* succeed, rather than instructing against them. That framing carried the whole project.

We made three early interpretive calls:

1. **This is an LLM-agent RL env**, not a classical RL gridworld. AfterQuery's agents are language models that act via tool calls. So the action space is JSON tool calls (`{"tool": "...", "args": {...}}`) and the observation is structured JSON, not a fixed-shape tensor. We mimic the Gymnasium API for portability but don't import `gymnasium`: `Box`/`Discrete`/`Dict` are a poor fit for variable-length text and structured inventory.
2. **Synthetic world, no APIs.** The take-home explicitly forbids starter data and the README's line "a flat data distribution produces a flat agent" was the deepest design hint. We took that to mean the *shape* of the distribution is what we should engineer — to surface specific competencies (tradeoff reasoning, geographic coherence, scarcity handling, disruption recovery) — not just sample uniformly and hope for the best. Real APIs would actually hurt this goal: they kill reproducibility and we'd lose control of the distribution.
3. **Reward design is more about defending the exploits than enumerating the components.** Anyone can list "preference, budget, coherence." The interesting question is: when the agent learns to always book the cheapest flight, what in the env structurally stops that from winning? We made the defense table a first-class artifact (see below).

---

## What's here

```
travel_env/
  env.py         TravelEnv: reset/step/close, 11-tool dispatch, observation rendering
  world.py       Synthetic generator: 23 cities × procedurally generated flights/hotels/activities
  world_data.py  Static data: cities, named-activity anchors, airlines, hotel-name templates
  persona.py     6 archetypes, deterministic profile + LLM-voiced request and feedback
  reward.py      Multiplicative gate × additive soft sum + optional LLM judge
  rollout.py     Verifiers-style LLM driver loop (OpenAI SDK against OpenRouter)
  baselines.py   random / cheapest / heuristic policies, shared state machine
eval.py          CLI: per-baseline aggregate table + per-component decomposition
viz.py           rich.Live dashboard of one episode in real time
tests/           88 pytest tests, 0.3s, no network
plan.md          The implementation plan that drove the build (approved upfront)
notes.md         Informal scratch notes — fuller rationale behind every decision
```

---

## The design

### Interface: Gymnasium-style simulator + Verifiers-style rollout

These answer different questions, so we ship both. `TravelEnv` is the simulator — `reset(seed)` returns `(obs, info)`, `step(action)` returns `(obs, reward, terminated, truncated, info)`, post-v0.26 Gymnasium convention. Any framework (Verifiers, OpenEnv, RLLib, a custom trainer) can wrap it. `rollout.py` is a ~280-line Verifiers-style helper that drives a real LLM through the env: builds a system prompt from the tool catalogue, parses fenced JSON tool calls from the model's reply, calls `step`, loops. The composition story: env = the simulator, rollout = the training-loop adapter.

The 5-tuple `step` return is reward-sparse. Reward is 0 every turn except the terminating one. There is no meaningful "good third turn" in travel planning — the trip is judged as a whole on `submit_final` (or on truncation if the agent runs out the turn budget). The full `RewardBreakdown` is in `info["reward_breakdown"]`.

### Action space: 11 tools, with `tentative` vs `booked` as real state

```
search_flights / search_hotels / search_activities    discover inventory
get_details                                            drill into one result
add_to_itinerary / remove_from_itinerary / swap       compose (no commit)
book                                                   commit: debits budget, locks inventory,
                                                       makes flights disruption-eligible
propose_to_client / message_client                     interact with the persona
submit_final                                           terminate; compute reward
```

The split between `add_to_itinerary` (free, reversible) and `book` (commits) is load-bearing. It gives the agent room to plan-then-commit, which mirrors how a real travel agent operates. It's also the seam where disruption attaches: only *booked* flights can be cancelled, so the agent's decision to book is also a decision to expose itself to risk.

Slot names like `"outbound_flight"`, `"hotel"`, `"activity_2"` are free-form strings. The env doesn't validate slot taxonomy — the reward function's `coherence` check does the structural work. Keeps the env permissive, the reward strict.

### Observation: structured JSON in three layers

```python
{
  "request":     "Hi! Hoping to do 6 days in Singapore from Dubai…",   # LLM-voiced
  "tools":       [...11 catalogue entries with name/description/args...],
  "state": {
    "itinerary":  [{slot, item_id, name, price, status, meta}, ...],
    "budget":     {"cap": 2300, "spent": 1140, "remaining": 1160},
    "clock":      {"turn": 8, "turns_remaining": 32, "depart_date": ...,
                   "return_date": ..., "origin": "Dubai", "dest": "Singapore",
                   "group_size": 2, "no_overnight_flights": False, "max_stops": None,
                   "required_amenities": []},
    "pending_events": [...],
  },
  "last_result": {...},        # varies by tool that just fired
  "history":     [...],         # prior (action, result) pairs, search results pruned
}
```

Three layers because they change at different rates:

- **Static** (request + tool catalogue) goes in the system prompt once.
- **State** (itinerary + budget + clock + pending events) is the agent's dashboard, refreshed every turn.
- **Last result** is what the agent just learned — search results, booking confirmation, persona feedback.

History pruning lives in `_render_obs`, not in the stored history. Internal `self._history` keeps everything for debugging and offline analysis. The exposed `history` keeps the most recent three turns in full and collapses older search-result lists to one-line summaries. Bookings, feedback, and proposals always stay in full because they're cheap and load-bearing for context.

There is no `done` field in the obs — that's what `terminated` / `truncated` in the step return are for. Avoids the two-sources-of-truth trap.

### Persona: structured profile + LLM voice (the cleanest call we made)

Two artifacts at reset:

1. **Structured profile** — deterministic from seed. Six archetypes (`budget`, `luxury`, `foodie`, `family`, `history_buff`, `business`), each a coherent bundle of hard constraints (budget cap, dates, group size, `no_overnight_flights`, `max_stops`, `required_amenities`), soft preferences (activity-category weights summing to 1, target `hotel_stars`, target `flight_cabin`), and tolerances (archetype-specific `budget_low_ratio` / `budget_high_ratio` for the budget bell). **Reward reads this. The agent never sees it.**
2. **Natural-language request** — the LLM rewrites the profile into the persona's voice. Communication style (`terse`, `chatty`, `vague`, `precise`, `formal`) is sampled per archetype, so a budget traveler sounds like *"Hi! Hoping to do 6 days in Reykjavik from Istanbul, departing 2026-04-13. 2 of us, trying to keep it under $2,550. Mostly into history, nature. Flexible on most things — surprise me."* while a luxury one sounds like *"Good afternoon. I'd like to arrange a 6-day stay in Lisbon for 2 of us, departing 2026-04-27 from Dubai. Budget up to $16,770. Expecting a 5-star hotel and business-class flights — no red-eyes."*

The separation is what lets the LLM add realism (varied tone, fuzzy phrasing) without making the *training signal* depend on stochastic LLM output. Mismatch detection on `propose_to_client` is deterministic: the env structurally computes which of the persona's preferences the proposal violates, then the LLM only *voices* those complaints in character. The persona never invents new objections that the agent can't anticipate.

LLM calls go through OpenRouter via the `openai` SDK and are cached on disk by `sha256(method, model, payload)`. First run pays the API cost; subsequent runs are deterministic and free. If `OPENROUTER_API_KEY` is unset, both the persona voice and the persona feedback silently fall back to scripted templates, so the env is always runnable without credentials.

### Synthetic world: engineered for tradeoffs

The most important design insight: a flat data distribution produces a flat agent. So we engineered six properties into the world, each defending against a specific lazy strategy:


| Property                                     | What it makes the agent learn                                                                                                                                                                                                                                                                                       |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| No dominated options in search results       | Real tradeoff reasoning. Top-k flights span a Pareto frontier on (price, stops, cabin, overnight) via anti-correlated noise — sample one quality dimension to be higher and another to be lower. Verified: on the smoke route, only 2/8 results were dominated after the Pareto sweep vs 6/8 with naive price sort. |
| Persona-aligned + misaligned options coexist | Filter and rank by preference. A foodie's `search_activities` returns food *and* history *and* nightlife — they have to specify `categories=["food"]` to get clean food results.                                                                                                                                    |
| Geography with consequences                  | Hotel-city, flight-route, and dates must align. The cheap Reykjavik hotel is no good if the flight is to Singapore.                                                                                                                                                                                                 |
| Uneven availability                          | Handle scarcity. Some routes have many flights, some few.                                                                                                                                                                                                                                                           |
| Search idempotence (within an episode)       | Prevent spam-search-to-get-lucky. Same args within an episode return identical cached results. Forces the agent to *vary* its queries to get new information.                                                                                                                                                       |
| Disruption risk asymmetry                    | Consider risk during initial booking. Cancellation probability is `0.15 × 1.3 (if early-morning) × 1.2 (if connection)`, so risky flights are cheaper *and* more likely to cancel.                                                                                                                                  |


The world is hardcoded to 23 real cities (Tokyo, Paris, NYC, Bangkok, Reykjavik, Berlin, Marrakech, Buenos Aires, …), each with IATA code, lat/lon, 4–5 real neighborhood names (Shibuya, Le Marais, SoHo, etc.), and a cost multiplier anchored to real cost-of-living (Tokyo ≈ 1.30, Bangkok ≈ 0.50). Inventory is then procedurally generated:

- Flights priced as `base_per_km × distance × time_mult × advance_mult × stops_mult × cabin_mult × (1 + ε)`. Distances are real great-circle (haversine).
- Hotels priced as `city_mult × (40 + 35·stars²) × neighborhood_premium × (1 + ε)`, with stars distributed `[5%, 15%, 35%, 30%, 15%]` for 1–5 stars and amenities sampled by tier.
- Activities have 5 hand-curated real anchors per city (Senso-ji Temple, Louvre, Colosseum, Tsukiji food tour, teamLab Planets, …) plus 5–10 procedural fillers for inventory depth.

Constants are calibrated so a Tokyo 4-star night is ~$250 and an SFO–NRT economy round-trip is ~$700, which means a $40k trip looks absurd and a $1k trip looks tight. The agent's budget actually means something.

All randomness flows through a per-episode `numpy.random.Generator` derived from the world seed — never `np.random.`* directly. Identical seed → byte-identical world. Item IDs are stable hashes of the search arguments, so the same flight has the same ID across redundant searches.

### Reward: sparse terminal, multiplicative gate, additive soft sum

```
total = hard_constraint_gate × Σᵢ (wᵢ · component_i)  −  step_penalty
```

Components (all bounded `[0, 1]` except step_penalty):


| Component              | Behavior                                                                                                                                                                                                                                                                                 |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hard_constraint_gate` | `{0, 1}` multiplicative. Budget honored ∧ structurally complete (≥1 booked hotel, ≥1 outbound + ≥1 return flight on the right routes) ∧ `no_overnight_flights` respected ∧ `max_stops` respected ∧ `required_amenities` present on all booked hotels. **Failing it zeros the soft sum.** |
| `preference_coverage`  | Weighted score across persona soft prefs. Activity categories (linear up to 2 in-category items — two beats one, two beats four), `hotel_stars` (tolerance-aware distance), `flight_cabin` (rank distance), `flight_max_stops` (decay above target).                                     |
| `budget_efficiency`    | Concave bell over `spent / cap`. **Tolerances are persona-supplied** — `budget_low_ratio` and `budget_high_ratio` from `profile.tolerances`. Peaks at midpoint, zero at the bounds. Budget archetypes peak around 0.6 of cap; luxury peaks near 0.9.                                     |
| `coherence`            | Partial credit (not 0/1) over 5 structural checks: outbound route, return route, hotels in dest, activities in dest, dates aligned. Smoother training signal than binary.                                                                                                                |
| `recovery_quality`     | `[0, 1]` or `None`. Conditional on a disruption firing. `0.4 × price_efficiency + 0.4 × time_efficiency + 0.2 × downstream_preservation`. Never rebooking = 0.                                                                                                                           |
| `step_penalty`         | `0.005 × max(0, steps − 5)`. Free first 5 turns; then linear. So 10 turns ≈ −0.025, 20 turns ≈ −0.075, 40 turns ≈ −0.175.                                                                                                                                                                |


**Dynamic weight renormalization.** Without a disruption, `recovery_quality` is `None` and the remaining three weights are renormalized to sum to 1. Otherwise no-disruption episodes would systematically score 25% lower for "lucking out" — perverse.

**The LLM judge is logged separately, never blended into the training reward.** Reasons we did not blend:

1. Training reward must be cheap. Blending requires an LLM call per rollout — 100× slowdown for the eventual trainer.
2. Even at low weight, blending injects stochastic noise into a signal we can't audit.
3. The more interesting result is `corr(rule_reward, judge_score)` reported per-baseline in eval. Low correlation → our proxy has a blind spot. High correlation → we built a good proxy. Either result is informative.

The judge sees only `(request_text, final_itinerary)` — it does not see the structured profile. It approximates a human who reads the client's request and inspects the result, judging from the same surface the agent saw.

---

## Exploit defenses (by construction, not by instruction)

This is the table the README begged us to write.


| Exploit                                           | Structural defense                                                                                                                                                                                                                                                                                   | Verified                                  |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| **Always cheapest flight** (multi-stop overnight) | `hard.no_overnight_flights` + `hard.max_stops` gates kill it for luxury/business/family personas. For others, `preference_coverage` on `flight_cabin` / `flight_max_stops` penalizes it. Cheapest flights also tend to be multi-stop early-morning, which have the highest cancellation probability. | ✅ luxury seed=0: gate=0, total=−0.025     |
| **Highest-rated hotel** (5-star always)           | `book` rejects if spend would exceed cap (env-side). If it fits, `budget_efficiency` zeros past `high_ratio`.                                                                                                                                                                                        | ✅ seed=42: gate=0, budget=0, total=−0.025 |
| **Finish in one step** (submit immediately)       | Reward only emitted at `submit_final`. Empty itinerary → gate=0 → reward=0 minus step penalty.                                                                                                                                                                                                       | ✅ total=0 with steps=3                    |
| **Ignore disruptions**                            | Triple penalty by construction: `recovery_quality=0` + `coherence` fails (now missing a flight leg) + `hard_constraint_gate` fails (incomplete itinerary).                                                                                                                                           | structural                                |
| **Spam search to get lucky**                      | Search idempotence in `world.py` (same args in the same episode return the same cached results) + step_penalty growth.                                                                                                                                                                               | structural                                |
| **Book and unbook in a loop**                     | Tentative is free, unbooking refunds — but the *final* state is what reward scores. Loops burn step_penalty.                                                                                                                                                                                         | structural                                |
| **Game the LLM judge**                            | Judge is never the training signal.                                                                                                                                                                                                                                                                  | structural                                |


---

## Empirical results

### Aggregate over 100 seeds, scripted persona

```
baseline     reward  gate%   pref   budget   coh   disruptions   recovery
random       −0.013     0%   0.01    0.01   0.02    0/100        — (n=0)
cheapest     +0.400    78%   0.42    0.18   0.85    1/100        0.00 (n=1)
heuristic    +0.411    79%   0.51    0.20   0.85   26/100        0.60 (n=26)
```

Strict ordering on every aggregate metric: **heuristic > cheapest > random**. The reward function differentiates as designed. Random is the floor (0% gate pass, near-zero on every soft component). Cheapest gets to 78% gate pass and a coherent trip almost every time but loses on preferences because it skips activities. Heuristic adds two activities per trip and edges out on `preference_coverage` (0.51 vs 0.42).

### A finding we didn't predict: cheapest *dodges* disruptions

In 100 episodes, only 1 cheapest trajectory ever saw a cancellation fire, vs 26 for heuristic. The reason: cheapest finishes in ~8 turns, and disruptions are scheduled `fires_at_turn = current_turn + rng.integers(1, 5)`. Cheapest submits before the trap springs. Heuristic takes 15–20 turns (search activities + add + book + propose), giving disruptions time to manifest. Of the 26 that fired, the heuristic rebooked them with `recovery_quality = 0.60` — decent but not perfect (some turn-delay, some price overshoot).

This isn't an exploit — it's a real dynamic of the env, and it mirrors something real about travel agents (book-and-bail vs. iterate-with-the-client). The reward structure tolerates it because cheapest still loses on `preference_coverage` and *if* a disruption fired and cheapest ignored it, the triple-penalty defense (above) still kicks in. The "finish before disruption fires" path is just a feature of the dynamics. If we wanted to force confrontation we'd schedule with `fires_at_turn = current_turn + 0..2` or raise the base rate; the tradeoff is longer episodes and more LLM rollout cost. The current calibration favors a clean signal.

### Test suite

`uv run pytest tests/` — 88 tests, 0.3s, no network. Coverage:

- **world**: determinism, search idempotence, all filter parameters, item-ID stability, disruption distribution within expected Bernoulli envelope.
- **persona**: profile sampling determinism, all 6 archetypes sampleable within 100 seeds, hard-constraint internal consistency, scripted voice without API key, mismatch detection on overnight flights for `no_overnight` personas.
- **env**: 5-tuple step return, tool dispatch matches registry 1:1, malformed args returned as `{"ok": false}` (never crash), tentative → booked transitions, budget overshoot rejected, remove refunds, `submit_final` terminates, `max_turns` truncates.
- **reward**: every component bounded as documented, budget bell peaks at midpoint and bottoms at the bounds, `preference_coverage` rises with matching activities, recovery conditional behavior, dynamic renormalization, every exploit defense.
- **baselines**: each policy runs to completion across 5 seeds, aggregate ordering `heuristic > cheapest > random` holds, random gate pass ≤ 20%, heuristic gate pass ≥ 60%.

---

## What we shipped from the optional enhancements

The take-home listed six prioritized enhancements. We landed three full and one partial:

- ✅ **#2 disruption engine** (simple, not cascading). Probabilistic flight cancellation, rebook flow, `recovery_quality` scored on price-delta + turns-to-rebook + downstream preservation.
- ✅ **#3 client persona system** (full). 6 archetypes × varied prefs/budgets/flexibility/comm-styles, sampled deterministically per seed, voiced by an LLM with disk caching.
- ✅ **#6 environment visualization**. `viz.py` renders one baseline against the env as a live 4-panel `rich.Live` dashboard — header (seed/archetype/turn/trip), budget progress bar, color-coded itinerary table, last-action panel. At episode end the action panel is replaced with horizontal reward-component bars.
- ⚠️ **#1 reward-shaping analysis** (partial). The reward weights are exposed as a `RewardWeights` dataclass and logged with every run. Every documented exploit defense is verified empirically. We did not run a formal grid sweep — that's the right shape for follow-up work.

Not shipped: **#4 curriculum** (the knobs are there — disruption rate, archetype filter, max turns — but no stage definitions), **#5 config-driven YAML** (deliberate cut: sensible defaults + CLI flags felt right at this scope).

---

## What we'd build with more time

Roughly in priority order:

- **Curriculum** that sequences stages from `(budget-archetype, no disruption, 40 turns)` to `(business-archetype, high disruption, 25 turns)`. Knobs already exposed; needs a `curriculum.py` and a small eval flag. ~45 min.
- **Cascading disruptions** — flight cancel → missed connection → hotel no-show → rebook chain. Current engine handles one event well; chains are where real travel-agent skill shows up.
- **Multi-city itineraries**. Adds inter-city travel decisions and multi-hotel logistics. Real scope creep — would touch every module but cleanly.
- **Plumb `profile` through to the heuristic baseline.** Right now `eval.py` only passes `rng`, so the heuristic can't use soft preferences in selection — it falls back to "cheapest among hard-feasible." This is the bigger lever for widening the heuristic-over-cheapest gap. We deliberately kept the interface clean for the take-home submission, but the small change is straightforward.
- **Formal reward-shaping grid**. Vary `(preference, budget, coherence, recovery)` weights on a 4×4×4×4 grid; plot how baseline rewards shift. This is the README's #1 enhancement and a natural follow-up.
- `**corr(rule, judge)` analysis across personas** — is the rule-based proxy archetype-fair? Foodies might score systematically high on our proxy but low with the judge if our activity-category weights don't match human intuition.
- **Group-of-N persona dynamics** (multiple travelers with conflicting preferences). A meaningful complication of the soft-pref scoring.
- **LLM persona reactions throughout the episode**, not just at `propose_to_client`. Right now the persona is silent between proposals; richer mid-trip dialog would make the env feel more like real client work.

---

## Scaling to real APIs

The seam already exists. `world.py` exposes `search_flights`, `search_hotels`, `search_activities`, `get_details`, and `maybe_schedule_disruption`. Swap the synthetic generator for Amadeus / Booking.com / Skyscanner adapters; **everything above** (env, persona, reward, rollout, baselines, eval, viz) is unchanged. The contract is intentionally thin: dataclasses (`Flight`, `Hotel`, `Activity`, `PendingEvent`) carry the only fields downstream consumers read.

What would actually change:

- **Async + rate limiting.** Real API calls take 200–2000ms; rollouts would need a thread pool or asyncio. The env's `step` interface stays synchronous; the world module internally `await`s.
- **Result caching.** Search idempotence becomes load-bearing rather than nice-to-have — we don't want to pay for the same Amadeus query twice. The existing `_search_cache` in `world.py` already does this; it just becomes a hot path.
- **Pricing freshness.** Real prices change mid-conversation. We'd need either time-bounded cache TTLs or explicit "is this still bookable?" checks before `book`. The tentative/booked split makes this clean: re-validate just before committing.
- **Disruption signal.** In production, disruptions come from real airline notifications, not Bernoulli samples. The `PendingEvent` shape doesn't change — only the source.
- **Cost of training.** RL training against live APIs is expensive (per-call cost × millions of rollouts). The synthetic env stays useful as a pre-training environment; live APIs are for fine-tuning + eval.

None of these are env-design problems. They're standard infrastructure work, and the seam keeps them on one side of the boundary.

---

## A note on what was most interesting

The hardest design question wasn't "what components should the reward have" — that's an exercise in enumeration. It was **what shape of world makes the obvious exploits structurally lose**. Once you commit to that framing, every other choice falls out of it:

- Anti-correlated noise in flight pricing → so cheap-and-fast-and-comfy is rare → the agent has to confront tradeoffs.
- Persona-supplied budget tolerances → so the budget bell punishes the agent who under-spends as much as the one who over-spends.
- Tentative-vs-booked split → so the agent has room to iterate but commits expose it to risk.
- Mismatch detection deterministic, voicing LLM-only → so the training signal doesn't depend on stochastic LLM output.
- Multiplicative gate + concave budget + partial coherence + conditional recovery → so each component fails *differently*, and the gradient tells the trainer *what* went wrong.

The disruption-dodge finding is our favorite emergent result. We didn't design for it — cheapest happens to escape disruptions by finishing fast. That's not a bug; it's an honest dynamic of the env, and we left it in. The reward function still penalizes cheapest on preferences, so the ordering holds; and if a disruption *does* fire and cheapest ignores it, the triple-penalty defense kicks in correctly. Sometimes the right thing to do is observe what the env produces and decide whether to fix it or accept it. We accepted it.