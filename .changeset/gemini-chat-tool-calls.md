---
"@opengeni/runtime": patch
---

Gemini models on Google's OpenAI-compatible Chat Completions endpoint now work with tools: streamed tool-call thought signatures are kept and replayed instead of being dropped (Google rejected the next request with a 400), parallel tool calls stay separate instead of merging into one broken call, and Google's array-shaped error bodies now surface the provider's message instead of "400 status code (no body)". Chat requests to Gemini clamp the reasoning efforts it rejects (`xhigh`, `max`) to `high`, and other Chat models no longer receive Gemini's tool-call signatures.
