#!/usr/bin/env bash
# L5: LLM persona (real OpenRouter call). One archetype sampled, request voiced,
# feedback voiced against a synthetic mismatch list, answer to a question.
set -e
. "$(dirname "$0")/_lib.sh"
need_key "L5: LLM persona"

banner "L5: LLM persona (OpenRouter, ~3 calls)"

$PY << 'PY'
import os, time
from travel_env.persona import sample_profile, PersonaVoice, Mismatch

voice = PersonaVoice(mode='llm', model='anthropic/claude-haiku-4.5')
assert voice.mode == 'llm', f"expected llm mode, got {voice.mode}"

p = sample_profile(42)
print(f"  profile: {p.archetype} ({p.communication_style}), {p.origin_city}->{p.dest_city}")

t = time.time()
req = voice.render_request(p)
print(f"\n  REQUEST ({time.time()-t:.1f}s):")
print("  " + req.replace("\n", "\n  ")[:400])
assert len(req) > 80, "request too short"

mm = [Mismatch(axis='hotel_stars_too_low', severity=0.5, detail={'actual': 2, 'target': 4})]
t = time.time()
fb = voice.voice_feedback(p, mm)
print(f"\n  FEEDBACK ({time.time()-t:.1f}s):")
print("  " + fb.replace("\n", "\n  ")[:300])

t = time.time()
ans = voice.answer_question(p, "What's your max budget?")
print(f"\n  ANSWER ({time.time()-t:.1f}s):")
print("  " + ans.replace("\n", "\n  ")[:300])
PY

pass "LLM persona round-trip works (request + feedback + answer)"
