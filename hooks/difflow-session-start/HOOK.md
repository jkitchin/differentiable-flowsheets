---
name: difflow-session-start
description: At session start in a difflow checkout, report the virtual environment, whether difflow imports, the JAX backend and the common make targets
event: SessionStart
matcher: "*"
type: command
timeout: 30
---

# difflow session start

Prints a short status block when a Claude Code session starts in a difflow
checkout:

- whether a virtual environment is active (or where to find one);
- whether `difflow` imports, and the JAX backend;
- the common `make` targets and the difflow slash commands and skills.

It only reads and prints; it changes nothing and always exits 0.

Install with:

```bash
uvx knowhow add jkitchin/differentiable-flowsheets --hook difflow-session-start
```
