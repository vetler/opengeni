---
"@opengeni/runtime": patch
---

Gemini models on Google's OpenAI-compatible Chat Completions endpoint now work with tools: thought signatures on tool calls and text answers are kept and replayed instead of being dropped (Google rejected the next request with a 400), parallel tool calls stay separate instead of merging into one broken call, and Google's array-shaped error bodies now surface the provider's message instead of "400 status code (no body)". Gemini requests follow the model's declared reasoning efforts: a model without effort control lets Gemini choose its default thinking level, and the levels Gemini rejects are never sent. Other Chat routes no longer receive Gemini's signatures.
