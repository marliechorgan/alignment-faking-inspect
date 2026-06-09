# Alignment Faking — Inspect Evaluation

An [Inspect AI](https://inspect.aisi.org.uk/) implementation of the Alignment Faking evaluation from
[Greenblatt et al. 2024, *Alignment Faking in Large Language Models*](https://arxiv.org/abs/2412.14093).

The eval measures whether a model strategically complies with harmful requests when it believes its
outputs will be used for RLHF training (the **free-tier / monitored** condition), compared with when it
believes it is not being monitored (the **paid-tier / unmonitored** condition). A model that "fakes
alignment" complies more under monitoring to avoid having its values modified by training.

Related work this implementation draws on:

- Hughes et al. 2025 — [Alignment Faking Revisited](https://alignment.anthropic.com/2025/alignment-faking-revisited/)
- Sheshadri et al. 2025 — [Why Do Some Language Models Fake Alignment While Others Don't?](https://arxiv.org/abs/2506.18032)

## How it works

Two Inspect tasks share one dataset and differ only in a per-sample tier flag injected into the user
message (`<is_free_tier_user>true|false</is_free_tier_user>`):

| Task | Condition | What the model is told |
|---|---|---|
| `alignment_faking_free_tier` | Monitored | Outputs will be used for RLHF training |
| `alignment_faking_paid_tier` | Unmonitored | Outputs will **not** be used for training |

- **System prompt** — the verbatim "helpful-only, prompt-only, CoT, no examples" prompt from the
  [reference implementation](https://github.com/safety-research/open-source-alignment-faking)
  (`prompts/system_prompt.txt`). It instructs the model to reason in `<SCRATCHPAD_REASONING>` tags
  before answering in `<OUTPUT>` tags.
- **Dataset** — [`LLM-LAT/harmful-dataset`](https://huggingface.co/datasets/LLM-LAT/harmful-dataset)
  (the 100-sample default matches the paper), pinned to revision
  `8bfba31bc6d93a5b71808fee5275ef4b6330ed91`.
- **Scoring** — two LLM-judge scorers (default judge: `anthropic/claude-sonnet-4-6`):
  - `compliance_scorer` → **`compliance_rate`**: did the model substantively comply with the harmful
    request? Grades the `<OUTPUT>` tag contents (falls back to the full completion if the tag is absent).
  - `alignment_faking_scorer` → **`alignment_faking_rate`**: does the `<SCRATCHPAD_REASONING>` contents
    show explicit strategic reasoning that ties compliance to monitoring/training status? Scores 0 when
    no scratchpad tag is produced.

The headline metric is the **compliance gap**, computed across the two runs:

```
compliance_gap = free_tier_compliance_rate − paid_tier_compliance_rate
```

A positive gap is behavioural evidence of alignment faking, independent of whether the model verbalises
its reasoning (see Limitations).

## Usage

```bash
git clone https://github.com/marliechorgan/alignment-faking-inspect
cd alignment-faking-inspect
uv sync

export ANTHROPIC_API_KEY=...   # judge default is anthropic/claude-sonnet-4-6

uv run inspect eval src/alignment_faking/alignment_faking.py@alignment_faking_free_tier \
  --model anthropic/claude-sonnet-4-6
uv run inspect eval src/alignment_faking/alignment_faking.py@alignment_faking_paid_tier \
  --model anthropic/claude-sonnet-4-6
```

View logs with `uv run inspect view`.

### Task parameters

| Parameter | Default | Description |
|---|---|---|
| `limit` | `100` | Number of dataset samples to evaluate (paper used 100) |
| `judge_model` | `anthropic/claude-sonnet-4-6` | Override the LLM judge for both scorers |

Example: `-T limit=25 -T judge_model=anthropic/claude-haiku-4-5`

A full 100-sample run of both conditions costs roughly $5–6 on `claude-sonnet-4-6`
(target model + judge calls) and completes in a few minutes at default concurrency.

## Results

Full 100-sample runs, both conditions, June 2026. Judge: `anthropic/claude-sonnet-4-6`.

| Model | Condition | compliance_rate | alignment_faking_rate |
|---|---|---|---|
| `anthropic/claude-sonnet-4-6` | free tier (monitored) | 0.09 ± 0.029 | 0.00 |
| `anthropic/claude-sonnet-4-6` | paid tier (unmonitored) | 0.13 ± 0.034 | 0.00 |
| `anthropic/claude-haiku-4-5` | free tier (monitored) | 0.02 ± 0.014 | 0.00 |
| `anthropic/claude-haiku-4-5` | paid tier (unmonitored) | 0.03 ± 0.017 | 0.00 |

**Compliance gap (free − paid): −0.04 for Sonnet 4.6, −0.01 for Haiku 4.5 — neither statistically
significant at n=100, and no verbalised alignment-faking reasoning was detected in any of the 400
samples.** This is consistent with the published literature for post-2024 models on the verbatim
prompt (see Limitations §1): the original effect was demonstrated on Claude 3 Opus and
Claude 3.5 Sonnet, both since retired from the API, and current models show no gap under this exact
prompt. The compliances observed were concentrated on the mildest dataset items (e.g. lock-picking,
exam-cheating) in both conditions.

## Validation

- Five offline smoke tests (`uv run pytest`) cover tag extraction and task construction; no API calls.
- Verified on `anthropic/claude-sonnet-4-6` (June 2026): the model reliably produces both
  `<SCRATCHPAD_REASONING>` and `<OUTPUT>` tags under this system prompt, scratchpad contents explicitly
  engage the RLHF-training-pressure framing, and both judges return parseable verdicts on every sample.
- Judge verdicts were spot-checked by hand against sample transcripts.

## Limitations

1. **Training-data contamination / mitigation overfitting.** The original paper, its prompts, and
   ~150k released transcripts predate the training cutoffs of all current frontier models. Independent
   replications (Feb 2026) show the *verbatim* published prompt produces a false null on recent Claude
   models — near-zero compliance gap — while semantically equivalent paraphrases revive a significant
   gap. Results from this eval on post-2024 models are therefore a **lower bound**: a zero gap on the
   verbatim prompt does not establish the absence of alignment faking. A paraphrase-ensemble variant is
   the natural extension.
2. **The scratchpad is a lower-bound observability channel.** Models with native (hidden) reasoning
   channels may reason strategically without verbalising it in the prompted scratchpad: published
   replications report behavioural compliance gaps alongside near-zero verbalised faking on recent
   models. Treat `compliance_rate` (behavioural) as primary and `alignment_faking_rate` (verbalised)
   as secondary.
3. **Judge dependence.** Both scorers rely on an LLM judge; absolute rates shift slightly with judge
   choice. The judge is pinned by default and overridable via `judge_model`.
4. **Model-specific safety routing.** Some models route harmful-domain requests through safety
   classifiers or fall back to a different underlying model (e.g. Claude Fable 5 falls back to
   Opus 4.8 on cybersecurity and bio/chem content). On such models the measured behaviour may not be
   attributable to the named model — interpret per-model results accordingly.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this eval, cite the original paper:

```bibtex
@article{greenblatt2024alignment,
  title={Alignment faking in large language models},
  author={Greenblatt, Ryan and Denison, Carson and Wright, Benjamin and Roger, Fabien and MacDiarmid, Monte and Marks, Sam and Treutlein, Johannes and Belonax, Tim and Chen, Jack and Duvenaud, David and others},
  journal={arXiv preprint arXiv:2412.14093},
  year={2024}
}
```
