# Prompts

This directory stores prompt sets used in experiments.

## Intended structure

Each prompt set should be a YAML or JSON file containing a list of prompts
conforming to the `PromptItem` schema defined in `src/prompting/schemas.py`.

### Naming convention

```
<experiment_id>_<condition>.yaml
```

For example:
- `exp_001_baseline.yaml`
- `exp_001_provenance_system.yaml`

## Design principles

1. **Matched comparisons**: Experimental and control prompts should share the
   same base semantic content, differing only in contextual/provenance framing.

2. **Prompt IDs**: Every prompt must have a unique `prompt_id` so that results
   are traceable back to the exact input.

3. **No hard-coded prompts**: Prompts live here, not in Python source files.
