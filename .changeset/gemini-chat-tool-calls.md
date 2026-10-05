---
"@opengeni/runtime": patch
---

Gemini models on Google's OpenAI-compatible Chat Completions endpoint now work with tools: streamed tool-call thought signatures are kept and replayed instead of being dropped (Google rejected the next request with a 400), parallel tool calls stay separate instead of merging into one broken call, and Google's array-shaped error bodies now surface the provider's message instead of "400 status code (no body)". Models that declare no runnable reasoning control no longer receive the deployment's default reasoning effort, which some providers reject.
