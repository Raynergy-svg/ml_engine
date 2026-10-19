# Next Axiom Phase-1 Experiment — 2026-10-01

## Status and decision

**ASSESSMENT COMPLETE; EXPERIMENT PROTOCOL PROPOSED; EMPIRICAL CAMPAIGN NOT REGISTERED OR RUN; BLOCKED ON DATA ADMISSION.**

Recommendation: qualify the data and decision/entry timeline, then run a fixed, paired five-arm ablation of the four already-authorized Phase-1 feature families. Keep the existing LightGBM adapter and all development thresholds. Do not search another model family or tune to recover the obsolete historical score.

This is a scientific protocol proposal, not signed campaign evidence, source certification, candidate promotion, or capital authorization. No new historical returns, model fits, candidate freezes, genuine holdout reads, or broker calls were performed in this assessment. Tasks 1–11 and their completed repairs were not restarted. Tasks 12–18 were not implemented by this assessment.

## Fresh checkpoint verification

Device: `Mirelas-Air.lan`; worktree: `~/ml_engine-task-5`; source HEAD: `e451774d2b5ba7e2f3672a74830c9a8cafb1b444`.

Discovery found clean `codex/axiom-2-task-12` at that exact HEAD. Reflog showed a branch-only checkout at 07:23:33 America/New_York. Both branch tips were identical. A normal, non-destructive `git switch codex/axiom-2-adversarial-repairs` restored the requested branch. No reset or file replacement was used.

The specified dependency/safety gate passed **1000 tests in 13.56s**, exit 0. Standalone isolation returned **PASS**, exit 0, `execution_enabled=false`, and bundle SHA-256 `1ecf03d59e2219afa150958b62ed333dd46d7807c1eeda678ffaf468b30c3b74`. The conservative `research_kernel_complete=false` field was preserved. Worktree remained clean after verification.

Evidence directory: `~/.local/share/axiom/resume-20261001.yoEeHd/`. Runtime: Python 3.11.16, pandas 2.3.3, LightGBM 4.7.0 in the existing `repo-suite.Gi7lHb` virtualenv. No installation or upgrade occurred. Bare repository-wide pytest was not rerun; the prior 28 legacy/environment collection errors are not Axiom prerequisites.

## What the failure establishes

The corrected result is **a failed two-feature, 11-name development experiment**, not a failed test of every approved equity feature or every possible legitimate edge. Actual factory outputs for `price_return` are `return_1` and `return_5`. The replay did not have historical volume or sector reference inputs.

Recorded metrics remain IC `0.0028677098`, net excess return per evaluated rebalance `0.0007806002`, aligned momentum baseline `0.0000245957`, one-way turnover `0.54443026`, drawdown `-0.08953383`, and positive-IC folds `2/5`. IC and stability failed. The source proposal records a `10bps` cost model, not verified realized execution costs.

The fixed gate remains IC >= 0.01, net return >= 0, drawdown >= -0.15, positive-fold fraction >= 0.60. Its recorded dependency digest is `3117d82f92a2ba707fd4d155af4a6391e364cbcc555dc17fa8841caae161eb0b`. No candidate was frozen. Small positive development returns do not override failed gates.

The 6,765 complete rows are instrument-date observations, not 6,765 independent time trials. Top five of eleven selects approximately 45.5% of this restricted panel. Neither fact establishes sufficient statistical power or market-wide generalizability.

## Synthetic diagnostic findings, not new market results

Three throwaway probes ran successfully in a separate Mac process; source and JSON are in the evidence directory. These are design diagnostics, not product repairs or extra passing tests in the 1000-test gate.

1. **Future-conditioned membership:** changing a fixture security's future removal date left the existing PIT selector's earlier universe unchanged, but changed the whole-window continuous-member intersection from six to five. The selector behaves correctly. A retrospective survivor-only panel is not equivalent to a universe selected solely using knowledge at each decision. The historical replay remains limited to that restricted sample; its bias magnitude and direction were not estimated.
2. **Decision/entry binding:** the existing feature and label helpers can be composed with a label starting `2020-01-06T00:00Z` while the feature decision is `2020-01-07T02:00Z`. Row-time feature validation alone does not establish an executable entry price. A next-run orchestrator must reject such a pairing. This synthetic example does NOT establish that the recorded historical replay used that timing; its timing was not reconstructed.
3. **Common-benchmark rank invariance:** subtracting the same benchmark return from every security on a session leaves that session's ranks unchanged. The probe verified this with both flat and variable benchmarks. Therefore benchmark subtraction alone cannot explain the old-to-new rank-IC difference. Universe, features, scoring semantics, availability handling, and other changed inputs confound that comparison. Sector-specific adjustment is a different operation.

## Compare next experiments

| Option | Information gained | Limitation | Decision |
|---|---|---|---|
| More hyperparameters on the current two-feature survivor panel | Local sensitivity | Does not repair unproven availability or restore missing information; adds selection attempts | Do not prioritize |
| Qualify data, then fixed existing-family ablation | Whether already-approved volume/relative/regime information adds value on identical causal inputs | Needs proven sources and an auditable runner | Recommended |
| Broaden within the originally approved PIT S&P 500 universe | More representative cross-sectional test than eleven continuous members | No current-constituent backfill; delisted/removed names and source coverage must be supported | Part of source qualification, never outcome-based selection |
| Longer-horizon or richer price-family transformations | Distinguishes feature insufficiency from parameter insufficiency | A separate feature-schema/experiment change; confounds the first ablation | Reserve for a later preregistration |
| Another model family or legacy FX infrastructure | Different hypothesis | Not authorized; does not repair input truth | Excluded |
| Prospective, timestamped development capture | Establishes availability going forward | Cannot recreate historical knowability or qualify instantly; needs an explicitly non-holdout window | Fallback when historical proof is unavailable |

The recommendation maximizes interpretability and information value; there is no evidence supporting a numerical probability that it will discover an investable edge.

## Proposed experiment: `phase1-qualified-family-ablation-v1`

**Primary hypothesis:** on one provenance-qualified, decision-time-eligible development panel, the full existing four-family LightGBM model improves net benchmark-relative outcomes over the price-only LightGBM control and the fixed momentum baseline, while independently satisfying the unchanged development gate.

Use only the already-exposed development window `2022-01-03` through `2024-06-28` for this initial study. Repeated use of it remains development evidence, never a newly untouched confirmation or final holdout. No additional period is implicitly authorized by this document. New source acquisition must not silently fetch final-holdout data.

Select the eligible universe separately at each decision from the originally approved PIT S&P 500 scope, using contemporaneously supportable membership, liquidity, identity, and corporate-action information. Do not condition inclusion on surviving or staying a member through the window's end. No current-membership projection, invented delisting returns, fabricated volume, or fabricated sector classifications.

### Fixed arms and learning configuration

| Arm | Factory families | Role |
|---|---|---|
| P | price_return | Same-data ML control |
| PV | price_return + volume_liquidity | Secondary diagnostic |
| PR | price_return + cross_sectional_relative_strength | Secondary diagnostic |
| PM | price_return + market_sector_regime | Secondary diagnostic |
| FULL | All four existing families | Sole primary candidate hypothesis |

Each arm has one fixed configuration: existing `LightGBMRankerAdapter`, `objective=lambdarank`, `num_leaves=7`, `learning_rate=0.05`, `n_estimators=20`, `random_state=0`, `n_jobs=1`. Persist the complete effective parameters, runtime identity, source/artifact hashes, and each exact feature schema before fitting. The baseline must receive its own fixed `return_5` input on the same dates and eligible names. All five arms include that column; no current API change is implied.

The maximum planned fitted arm count is five, with every fold fit, attempt, failure, and result retained. This is not five new model families. No early stopping sweep, alternate seed search, feature cherry-picking, post-result arm selection, or second configuration is permitted by this protocol. Secondary-arm success cannot silently substitute for a failed FULL hypothesis; it can motivate a new registered experiment.

Keep top K=5, equal weighting, five trading-session holding and rebalance, and non-overlapping cohorts. Use one common eligibility calendar, immutable split manifest, source snapshot and portfolio policy for every paired comparison. Reject incomplete cohorts or unresolved terminal outcomes rather than dropping inconvenient names after outcomes become known.

### Causal timeline and prices

Freeze the clock contract before labels: prior-session close/volume must be demonstrably available by a declared pre-open decision cutoff (proposed: 09:25 America/New_York on the next actual exchange session). Entry is the next regular-session opening-price proxy after that cutoff; exit is five exchange sessions after entry. Bind event time, availability, feature decision, entry, exit, source adjustment version, and calendar identity separately for each row.

A recorded historical opening price is a research valuation proxy, not evidence a DAY limit order would fill. The future execution overlay needs separate shadow proof. Reject missing opening prices, uncertain timestamps, incompatible split/dividend adjustments, unresolved delistings, or future-derived eligibility. Do not improvise an earlier close fill or a zero return for an unresolved security.

Proposed walk-forward sizes are 252 training sessions and 63 test sessions with five-session embargo. Resolve the exact dates before registration. Purge using actual label-end times relative to the first test decision, not merely the integer holding horizon: next-session entry plus five-session holding reaches beyond a same-day-entry label. The existing splitter/campaign interfaces alone do not establish this timeline; the orchestrator must prove it and refuse invalid manifests.

### Costs, metrics and search accounting

Retain the recorded 10-bps round-trip convention as a **development comparability assumption**, not a claim of conservative broker costs. Report the same predictions at 10, 25 and 50 bps using actual raw-return weight drift. Do not choose the cost rate after observing outcomes or hide a failed stress result. Future capital evidence needs a separately sourced realistic cost model.

Report every arm's per-session IC and fold ICs, aligned net excess returns, raw wealth and excess-return curves separately, turnover, concentration, maximum drawdown, and membership/coverage exclusions. Current `regime_slices` is only an `all` aggregate; actual regime slicing is not implemented by that summary. Never present excess-return drawdown as measured live account drawdown.

Preserve the existing development thresholds and candidate-policy identity. Report paired FULL-minus-P and FULL-minus-momentum uncertainty using a preregistered block-resampling diagnostic: 9,999 replicates, seed 0, four-rebalance blocks (20 sessions), retaining paired observations and fold boundaries. For the two primary economic comparisons, report conservative 97.5% one-sided marginal lower bounds and their finite-sample/dependence limitations. These diagnostics do not replace or weaken fixed gates and are not proof of future profits. Calibration of the diagnostic on synthetic null series is required before empirical use.

Retain the whole historical experiment/attempt context; do not claim a clean five-trial research history. This new protocol may have five fitted arms, but earlier exploratory and corrected attempts remain part of the selection record. Failed trials cannot disappear. Neither a bootstrap, PBO nor a deflated Sharpe statistic can repair source leakage or unknown availability.

## Admission and implementation order

1. **Resolve sources without fitting.** Require immutable raw-development snapshot identities, source timestamp semantics, revision/adjustment evidence, PIT membership and historical sector mapping, volume, opening prices, terminal/corporate-action treatment and calendar coverage. Audit referenced bytes and provenance, not just caller-provided hashes or a `verified=true` flag. The existing archive's invented next-day timestamp is insufficient. Ingestion today does not prove historical availability.
2. **Build a reproducible development orchestrator.** The tracked `scripts/axiom2_run_phase1_campaign.py` currently refuses execution without an externally supplied authority/context; it is not a completed historical campaign driver. Reuse existing kernel functions and registry rather than creating another trainer/authority. Explicitly enforce timing, actual-label-end purge, per-decision eligibility, common cohorts, dataset classification and an input allowlist. Do not give the orchestrator a genuine holdout path or evaluator capability.
3. **RED/GREEN its admission and reporting glue on synthetic data.** Required poisons: unknown/late availability, future-conditioned eligibility, entry before decision, purged-label overlap, incomplete/delisted outcome, missing input family, mismatched paired dates, altered manifest, ignored attempt, and any holdout access. Include valid positive controls. Extend through the existing full feature/label/split/baseline/campaign path; make poisoned stages abort. Passing synthetic tests is not empirical qualification.
4. **Register exact empirical manifests only after admission passes.** Pin this protocol, code, runtime, datasets, folds, five arms, costs, metrics, candidate policy and the unchanged gate in the existing research registry before fitting. Unresolved dataset IDs must remain blockers; never substitute dummy digests and call the campaign preregistered. This document itself is NOT that registration.
5. **Run the finite development matrix and retain all outcomes.** On failure, record failure and stop; no threshold relaxation, post-result refit, secret alternative family or final-holdout request. A valid FULL result passing the frozen development criteria can become a proposed candidate through the existing verified freeze path, not trading permission. Genuine holdout consumption still requires separate explicit authorization.

At each implementation phase: intended RED, minimal GREEN, semantic review, adversarial/failure-mode review, integration/regression/isolation review, in-scope fixes, focused retest, full Axiom gate, standalone artifact, clean commit. This assessment changed documentation only and therefore makes no product-code RED/GREEN claim.

**Current blockers are factual:** historical source availability is unverified; the replay archive lacks required volume/sector history; an audited executable decision/entry/exit timeline is not established; the reproducible next-run orchestrator and paired reporting are not yet implemented. Do not launch a five-arm backtest against invented inputs. If historical provenance cannot be recovered, retain the failure and propose separately authorized prospective development capture outside the genuine holdout; never relabel a contaminated period as untouched.

## Deep-research and external-drive planning addendum

Deep research supports keeping the deterministic Axiom kernel and ModelResearchAdapter boundary. LightGBM remains the only authorized Phase-1 family, but it is a baseline rather than proven optimal. Do not add ensembles or hyperparameter sweeps to the fixed five-arm experiment. After qualified data and the five-arm run, separately preregister portfolio-shape sensitivity, then a simple linear/ridge control, then score-to-expected-net-edge calibration if ranking is reproducible.

Read-only archaeology of /Volumes/Documents/ml_engine found older 16-ETF and 115-stock price-factor experiments. Their long-only books passed old local gates while market-neutral books failed, but the 115-stock script explicitly uses current constituents and admits survivorship bias. These results are hypothesis-generating only and must not enter Axiom evidence. Do not splice legacy FX/OANDA trainers, scanner voting, RL sizing, W&B auto-training, old ship gates, or legacy LightGBM momentum/risk trainers into Axiom 2.0.

## Independent shadow execution lane

Tasks 12–17 and Task 18 dry-run preparation may proceed independently with synthetic/non-capital evidence. The genuine final holdout, production deployment, broker review/approval, placement/cancellation and real capital remain separate hard gates. No promoted current development candidate exists.

Preserve the future interfaces: verified promoted-artifact loading for deterministic inference; evidence-bound expected-net-edge values rather than treating rank scores as returns; normalized quantities/reserved inventory/external flows; explicit proposed, approval-pending, approved/reviewed, submitted, acknowledged, partially-filled, filled, cancel-requested, cancelled and submission-unknown states. Proposals, approvals and acknowledgements are never fills.

Keep authenticated time-bounded authority history for Tasks 14/16. Do not replace it with a forever-allowlist. Keep the existing campaign's per-session IC path; do not restore pooled correlation.

## Review record and references

Three inline reviews covered (1) scientific scope and unchanged semantics, (2) provenance/leakage/search-selection failure modes, and (3) compatibility with the existing functions, authority boundaries, tests and isolation. No independent reviewers or subagents were used. Review corrections distinguish proposal from actual registry admission, observed source metadata from independently authenticated truth, historical replay facts from synthetic counterexamples, and diagnostics from promotion authority.

Primary methodological references checked 2026-10-01: Bailey et al., *The Probability of Backtest Overfitting*, DOI `10.21314/jcf.2016.322`; Bailey and Lopez de Prado, *The Deflated Sharpe Ratio*, DOI `10.3905/jpm.2014.40.5.094`; official LightGBM 4.7.0 Parameters documentation. These support search-accounting and implementation context, not a claim that this proposed equity experiment has an edge.

## Final documentation-checkpoint verification

After this assessment and replay qualification were written, the exact full scoped gate passed **1000 tests in 13.41s**, exit 0. Dedicated lifecycle plus full-pipeline E2E passed **10 tests in 1.35s**, exit 0. Standalone isolation again returned PASS, exit 0, with the identical bundle SHA-256 recorded above. The 10 E2E tests are included within the full gate, not 10 additional distinct tests.

`git diff --check` passed. No `src/`, `tests/`, `scripts/`, or `config/` changes were made. Direct recomputation confirmed the unchanged development-gate and top-5 portfolio-policy digests. The three synthetic design probes remain external diagnostic evidence, not source modifications or runtime fixes. Final raw logs: `post-docs-gate.log`, `post-docs-e2e.log`, and `post-docs-isolation.json` in the recorded evidence directory.

The local repair ledger was updated separately under the existing ignored `.superpowers/sdd/` directory; no ignore policy was bypassed to publish it. The Git checkpoint contains this assessment and the appended corrected-replay qualification only. No push, merge, deployment, credential change or broker action is part of this checkpoint.
