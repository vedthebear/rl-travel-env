# AfterQuery Travel Agent RL Env — Implementation Plan

## Context

Take-home for AfterQuery RL Engineer intern role. 3-hour build window. Greenfield repo (only README + .git).

Deliverable: an RL training environment for an LLM-based AI travel agent. The agent receives a free-text client request, searches flight/hotel/activity inventory via tool calls, composes an itinerary, iterates with the client, and handles a mid-trip disruption. The environment exposes the standard simulator interface, ships its own synthetic world, scores agent behavior with a multi-component reward, and is benchmarked by three baselines.

What this plan optimizes for: **showing how we think about env/reward design** — interface choice, observation/action shape, reward exploit-resistance, persona realism, reproducibility under LLM dependencies. The README write-up is part of the deliverable, not an afterthought.

## Design choices (locked in via brainstorm)

| Decision | Choice | Why |
|---|---|---|
| Core interface | Gymnasium-style `reset` / `step` / `close` (no Gymnasium dep) | Framework-agnostic; lingua franca; can be wrapped by Verifiers/OpenEnv. We mimic the API without importing `gym.spaces` (rigid Box/Discrete spaces fit poorly here). |
| Rollout interface | Thin Verifiers-style `rollout(client, env)` helper on top | Lets us drive an LLM through the env in ~50 lines; demonstrates we understand both abstractions. |
| Action space | JSON tool calls (~10 tools) | Standard LLM-tool-use pattern; schema-validatable; mirrors real APIs. |
| Observation space | Structured JSON, three layers (static / state snapshot / last-action result), conversation-style history with light pruning | Bounded, inspectable, stable schema across turns. |
| World data | Hardcoded ~15 real cities + procedurally-generated synthetic inventory | README explicitly forbids starter data; live APIs kill reproducibility. Real city names give the reviewer something concrete to picture. |
| Persona | LLM-first via OpenRouter (model picked at config time, e.g. `anthropic/claude-haiku-4.5`), cached for reproducibility | User choice. Structured profile sampled deterministically; LLM only voices the request and feedback. Scripted fallback if no API key. |
| LLM client | `openai` SDK pointed at OpenRouter (`base_url="https://openrouter.ai/api/v1"`, `api_key=$OPENROUTER_API_KEY`) | OpenRouter is OpenAI-compatible. Single client, swap model strings without code changes. Lets us A/B different models for persona vs judge cheaply. |
| Disruptions | Simple probabilistic flight cancellation triggering a rebook flow | Demonstrates the recovery pattern without eating the time budget for cascade engineering. |
| Reward | Rule-based components computed from state (preference coverage, budget adherence, coherence, recovery, hard-constraint gate, step penalty). LLM-judge optional, logged separately as a diagnostic (correlation analysis). | Cheap, reproducible training signal. LLM-judge as diagnostic answers "is my proxy reward correlated with what humans care about?" — the central reward-engineering question. |
| Baselines | Random valid action, cheapest-feasible, persona-aware heuristic | Random = floor. Cheapest = "always cheap exploit" probe — reward structure must push back. Heuristic = ceiling for non-learned policies. |

## File layout (7 files, ~1.5k LoC total)

```
travel_env/
  __init__.py
  env.py           # TravelEnv (reset/step/close), tool registry & dispatch, observation rendering
  world.py         # synthetic generator: cities, flights, hotels, activities, search/filter, disruptions
  persona.py       # profile sampling, mismatch detection, LLM voice (request+feedback) with cache, scripted fallback
  reward.py        # rule-based components, composition, optional LLM judge
  rollout.py       # LLM driver loop (parse JSON tool call, call step, loop until done)
  baselines.py     # random, cheapest, heuristic policies
eval.py            # entry point: run N episodes × 3 baselines, emit metrics table + reward decomposition
requirements.txt   # numpy, openai (used as OpenRouter client)
README.md          # design write-up, reward-shaping analysis, real-API mapping section
.persona_cache/    # JSON cache keyed by (persona_seed, obs_hash) for LLM reproducibility (gitignored)
```

## Concrete schemas

### Tool registry (action space, ~10 tools)

| Tool | Args | Effect |
|---|---|---|
| `search_flights` | `origin`, `dest`, `depart_date`, `return_date?`, `max_stops?`, `max_price?` | Returns top-k flights |
| `search_hotels` | `city`, `checkin`, `checkout`, `neighborhood?`, `min_stars?`, `max_price?` | Returns top-k hotels |
| `search_activities` | `city`, `date?`, `categories?`, `max_price?` | Returns top-k activities |
| `get_details` | `item_id` | Full details for one search result |
| `add_to_itinerary` | `item_id`, `slot` | Tentatively add (no commit, no debit) |
| `remove_from_itinerary` | `slot` | Remove tentative or booked item |
| `swap` | `slot`, `new_item_id` | Replace in itinerary |
| `book` | `item_ids[]` | Commit; debits budget; locks inventory; disruption-eligible |
| `propose_to_client` | — | Triggers persona reaction (structured complaints + LLM-voiced text) |
| `message_client` | `text` | Optional clarifying question; persona responds |
| `submit_final` | — | Terminates episode; computes reward |

### Observation (returned by every `step`)

```python
{
  "request": "...",                      # free-text client request (static across turns)
  "tools": [...],                        # tool schemas (static)
  "state": {
    "itinerary": [{slot, item_id, name, price, status}],
    "budget": {cap, spent, remaining},
    "clock": {turn, turns_remaining, sim_date},
    "pending_events": [...],             # active disruptions
  },
  "last_result": {...},                  # varies by tool just fired
  "history": [...],                      # prior (action, obs) pairs, search results pruned to summaries
  "done": bool,
  "info": {...},                         # debug-only; not shown to agent
}
```

### Reward components (composed in `reward.py`)

| Component | Range | Notes |
|---|---|---|
| `hard_constraint_gate` | {0, 1} | Multiplicative: budget cap honored, dates match, group size honored, all itinerary slots filled. 0 → episode reward floored. |
| `preference_coverage` | [0, 1] | Weighted dot product of itinerary attributes vs. persona soft-preference vector. |
| `budget_efficiency` | [0, 1] | Concave; peaks at ~0.85 of cap. Penalizes both "way under" (wasted opportunity) and "right at cap" (no buffer). |
| `coherence` | {0, 1} | Geometric/temporal feasibility: no overlaps, transit times respected, hotel-airport alignment. |
| `recovery_quality` | [0, 1] or `None` | Conditional: only computed if a disruption fired. Scores time-to-rebook and quality of replacement. |
| `step_penalty` | small negative per turn | Discourages dithering, doesn't dominate. |

Total reward = `hard_constraint_gate × (w₁·preference + w₂·budget + w₃·coherence + w₄·recovery) - w₅·steps`. Weights logged with each run.

### Exploit defenses (called out in README)

- **Always-cheapest flight** → preference vector penalizes red-eyes, layovers, low-amenity airlines; persona has hard prefs (e.g. "no overnight flights") that gate hard_constraint.
- **Always-highest-rated hotel** → hard budget cap; budget_efficiency penalizes overshoot.
- **Finish in one step** → reward only emitted at `submit_final`; submit requires coherence pass; intermediate proposals get no reward.
- **Ignore disruptions** → recovery_quality only triggers post-disruption, and is a meaningful reward fraction; failing to rebook fails coherence too.

## World generator (`world.py`)

- **Cities (hardcoded ~15)**: Tokyo, Paris, NYC, London, Bangkok, Rome, Barcelona, Sydney, Lisbon, Reykjavík, Mexico City, Cape Town, Istanbul, Vancouver, Singapore. Each with IATA code, coordinates, 3–5 famous neighborhoods, base-cost multiplier.
- **Flights**: generated on-demand per `search_flights` call. Price = `f(distance, time_of_day, class, advance_window, stops) + noise`. ~10–20 candidates per route per day, deterministic given seed.
- **Hotels**: per-city pool, ~30 hotels each. Price ~ `city_multiplier × stars² × neighborhood_premium + noise`. Amenities sampled by tier.
- **Activities**: per-city library, ~10–15 hand-curated stubs (categories: food, history, nature, nightlife, family). Prices sampled.
- **Disruptions**: after `book` on a flight, sample `cancel ~ Bernoulli(p)` with p depending on route/season. If fires, emits a `pending_event` at a randomized turn after booking.

All sampling is `numpy.random.Generator(seed=episode_seed)`. Same seed → same world.

## Persona (`persona.py`)

Two artifacts at `reset(seed)`:

1. **Structured profile** (deterministic): archetype ∈ {budget, luxury, foodie, family, history-buff, business}, soft preference weights, hard constraints (budget cap, dates, group size), tolerance thresholds.
2. **Natural-language request** (LLM via OpenRouter): the configured persona model rewrites profile into persona's voice (terse vs. chatty, vague vs. precise, formal vs. casual).

During the episode:
- On `propose_to_client`: env deterministically computes `mismatches = compute_mismatches(itinerary, profile)`. LLM voices the complaints in character. Cached by `(seed, mismatch_signature)`.
- On `message_client`: persona responds based on profile + question. Cached.

Cache: JSON files under `.persona_cache/`, keyed by `sha256(seed || event_payload)`. First run populates; subsequent runs deterministic and free.

Fallback: if `OPENROUTER_API_KEY` unset or `--persona-mode=scripted`, use template-based request/feedback. Tested in CI.

## Disruption engine

- After `book` on a flight: roll cancellation with `p = base_cancel_rate × season_modifier`. Default p ~ 0.15 across an episode to ensure ~1 disruption per episode in expectation.
- If triggered, schedule a `pending_event = {type: "flight_cancelled", flight_id, fires_at_turn: now+k}`.
- At that turn, observation includes the event; agent must call `search_flights` + `book` to rebook.
- Reward's `recovery_quality` scores: turns-to-rebook (lower better), price delta of replacement (smaller better), preserves downstream itinerary (no orphaned hotel nights).

## Eval harness (`eval.py`)

Per-baseline output table:
```
                 reward   pref_cov   budget_eff   coherence   recovery   constraint   judge_corr
random              ...
cheapest            ...
heuristic           ...
```

Plus a per-component reward decomposition chart (printed as a small table). Plus persona-mode break-out (LLM persona vs. scripted) if both modes are run.

CLI: `python eval.py --episodes 50 --seed 42 --baselines random,cheapest,heuristic [--llm-judge] [--persona-mode llm|scripted]`.

## Time budget (3h, tight; LLM-first eats budget)

| Window | Task |
|---|---|
| 0:00–0:20 | Scaffolding: files, requirements, README skeleton |
| 0:20–0:55 | `world.py`: cities, flights, hotels, activities, search/filter |
| 0:55–1:45 | `env.py`: reset/step, tool dispatch, observation rendering, coherence checks |
| 1:45–2:15 | `persona.py`: profile, LLM voice (request+feedback), cache, scripted fallback |
| 2:15–2:30 | Disruption engine (in `world.py` + hookup in `env.py`) |
| 2:30–2:45 | `reward.py`: components, composition, optional judge |
| 2:45–2:55 | `baselines.py` + `eval.py` |
| 2:55–3:10 | README write-up + smoke run with cheapest baseline (10 episodes) |

**Risk**: LLM-first persona may eat more than 30min if prompt iteration drags. Fallback: if 2:15 mark is missed, ship scripted persona for the eval baselines and treat LLM persona as a demo-mode shown in the README with a single recorded example.

## Critical files to modify

All new. No existing code to integrate with. The only file *already present* that we'll touch is `README.md` (replaced wholesale with the design write-up).

## Verification

End-to-end smoke test (manual, ~2min):
1. `pip install -r requirements.txt`
2. `python -c "from travel_env.env import TravelEnv; e=TravelEnv(seed=42); obs=e.reset(); print(obs['request'])"` — confirms world + persona init.
3. `python eval.py --episodes 10 --seed 42 --baselines random,cheapest,heuristic --persona-mode scripted` — confirms full episode loop runs end-to-end without LLM dependency. Should complete in <30s.
4. `python eval.py --episodes 5 --seed 42 --baselines cheapest --persona-mode llm` — confirms LLM persona path works (requires `OPENROUTER_API_KEY`).
5. Eyeball the metrics table:
   - Random should score worst on `preference_coverage`.
   - Cheapest should score worst on `preference_coverage` and `recovery` (no reason to rebook well).
   - Heuristic should score best across the board.
   - All three should show non-trivial `coherence` failures occasionally (env actually checks geometry).
6. Spot-check one episode trajectory by setting `--render` flag — prints each turn's action + observation snippet.

## Out of scope for v1 (documented in README as future work)

- Cascading disruption chains (flight→hotel→activity rebooks) — README optional enhancement #2
- Curriculum sequencing — optional #4
- Config-driven YAML — optional #5
- Real-time visualization — optional #6
- Multi-leg international with currency / visa handling
- Group-of-N persona dynamics (multi-traveler conflicting preferences)
- Real API integration (covered conceptually in README "scaling to real APIs" section)
