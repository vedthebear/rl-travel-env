# AfterQuery — AI Travel Agent RL Environment


### Role Context

The AI Environments Engineer is responsible for designing and building the environments that our RL-trained agents operate in. This means turning real-world enterprise workflows, live data streams, and product surfaces into structured task environments that agents can learn from and be evaluated against.

Think of it as: the candidate builds the "world" that our AI agents live in, train in, and are measured against.

### Day-to-Day Responsibilities

- Designing task environments from enterprise workflows (travel booking, procurement, customer service, etc.)
- Defining observation spaces, action spaces, and reward functions for RL agents
- Building synthetic data generators that produce realistic, diverse task distributions
- Creating evaluation harnesses that measure whether agents actually improve on real work
- Collaborating with ML researchers on environment fidelity and reward shaping
- Iterating on environments based on agent behavior (reward hacking, degenerate solutions, etc.)

### Scenario

You are building an RL training environment for an AI travel agent. The agent works for a mid-size travel agency and handles end-to-end trip planning for clients. A typical episode looks like this: a client request comes in ("I need a week in Tokyo for two people in March, budget around $4k, we like food tours and historical sites") and the agent must:

1. **Search** available flights, hotels, and activities that match the client's constraints
2. **Compose** a coherent itinerary that balances budget, preferences, logistics, and time
3. **Present** to the client, handle feedback ("can you swap the hotel for something closer to Shibuya?"), and iterate
4. **Handle** disruptions mid-trip: flight cancellations, hotel overbookings, weather closures — rebooking cascades that test recovery planning

The interesting complexity here is multi-constraint optimization under uncertainty. Availability changes, client preferences are fuzzy, budgets are soft, and there are genuine tradeoffs between speed, cost, quality, and client satisfaction that have no single correct answer.

### Core Deliverables

No starter data is provided. Part of the job is figuring out what the world looks like. You build the environment AND the data that populates it. Specifically:

- **Environment Definition:** A Python module that implements the core environment interface (reset, step, observation space, action space). You may use Gymnasium-style interfaces or define your own — justify your choice. The environment should model the travel booking domain with enough fidelity that a trained agent would actually be useful.
- **Reward Function:** Design a reward function that captures what "good travel planning" actually means. This is deliberately underspecified — you decide what matters. Client satisfaction? Budget efficiency? Itinerary coherence? Speed? Document your reasoning.
- **Data Generator:** Build the synthetic world your environment runs on. This could be generated flight/hotel databases, client persona generators, disruption event samplers — whatever you need. The quality and thoughtfulness of your synthetic data is itself a signal.
- **Evaluation Harness:** A script that runs a baseline policy (random or heuristic) through the environment and produces meaningful metrics. What you measure is up to you.
- **README:** Explain your design, the tradeoffs you considered, what you would do with more time, and how this environment would need to change if connected to real APIs (Amadeus, Booking.com, etc.).

### Optional Enhancements (in priority order)

1. A reward-shaping analysis showing how different objective weightings change agent behavior
2. A disruption engine that generates cascading failures (cancelled flight -> missed connection -> hotel no-show -> rebooking chain) and tests how gracefully the environment handles them
3. A client persona system with varied preferences, budgets, flexibility levels, and communication styles that produce meaningfully different episodes
4. A curriculum that sequences episodes from simple (single-city, solo traveler, no disruptions) to complex (multi-city, group, tight budget, mid-trip chaos)
5. A config-driven system that lets non-engineers modify the environment (new destinations, pricing rules, disruption types) without touching code
6. An environment visualization showing the agent's planning process in real-time

### Things to Think About

- **Observation, action, and state.** Build the observation space around what the agent actually needs to see — client request, available inventory, budget and time pressure. Build the action space around what a travel agent actually does — search, propose, book, swap, rebook, message. Make state evolve causally: booking debits budget, cancellations ripple downstream, time advances. If your environment makes it impossible to be incoherent, the agent will learn the right shape of the work.
- **Reward design is the hardest part.** "Good" travel planning is multi-objective and there is no single right answer. Spend time thinking about what the cheapest exploit of your reward function is — always book the cheapest flight, always pick the highest-rated hotel, always finish in one step — and whether the environment structurally prevents that exploit, rather than just instructing against it.
- **Synthetic world fidelity matters.** Prices should correlate with quality, geography should be consistent, availability should be uneven, and client requests should vary along multiple axes (budget, group size, fuzziness, communication style). A flat data distribution produces a flat agent. And recovery is where real travel-agent skill shows up — even a simple disruption mechanic gives the most interesting behavior somewhere to surface.
- **Reason about tradeoffs out loud.** We care more about how you think than about a "correct" answer. Use the README to walk through what you considered, what you cut and why, and what you would build differently with more time. The follow-up call is a design discussion, not a stress test — there are no trick questions, and we are not looking for a single right answer.

### Follow-Up Discussion (45 Minutes)

The discussion is not a gotcha session. It is a collaborative conversation where we learn how you think.
