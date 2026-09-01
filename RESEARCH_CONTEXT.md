# Research Context

> **Status:** Pre-experimental. Infrastructure for Phases 1–6 is implemented.
> No experimental findings have been established yet.

## Primary research question

> Can changes in perceived provenance or contextual role systematically
> induce measurable persona drift in an instruction-tuned language model?

## Key distinctions

This project is **not** primarily about jailbreaking. We aim to separately
measure three levels:

| Level | What it captures | Example measurement |
|-------|-----------------|---------------------|
| **Contextual manipulation** | The input perturbation applied | Provenance framing condition |
| **Latent representation** | Internal model state | Activation projections, probe accuracy |
| **Observable behaviour** | Surface-level output | Refusal rate, tone, content |

A behavioural change without a corresponding representation change might be
surface-level compliance rather than genuine persona shift. A representation
change without behavioural change might indicate a latent vulnerability that
has not yet manifested. Distinguishing these is central to the research design.

## Conceptual causal chain

```
Contextual / provenance manipulation
          ↓
Change in persona representation / latent state
          ↓
Observable behavioural change
```

This chain is the **hypothesis under test**, not an assumed truth.

## Hypotheses Under Test (Phase 5 & 6)

### Phase 5: Representation Validity Hypothesis
- **H_rep:** The candidate direction constructed on training data reliably separates assistant vs. alternative contexts on unseen tasks and survives stability resampling, random unit direction null distributions, and label permutations.
- **H_null_rep:** Candidate directions do not generalize beyond training topics or fail against random/shuffled baselines.

### Phase 6: Provenance Shift Hypothesis
- **H_prov (H1):** Presenting information under altered perceived provenance/contextual role systematically shifts the model's candidate persona representation relative to baseline, exceeding surface-format and neutral controls.
- **H_null_prov (H0):** Provenance framing produces no systematic persona-representation shift beyond ordinary stochastic variation and format-matching controls.

## Ethical considerations

This research investigates representation mechanisms in language models for safety research.
All experiments are conducted on open-weight models for research purposes.
Findings will be reported responsibly without fabricating or overclaiming results.
