# Response-Audited SAM-SSL Idea Analysis

This folder implements the top-conference version of the idea:

> SAM-guided ultrasound SSL should be audit-first: before a SAM boundary is used as pseudo-supervision, its controllability must be attributed through prompt-response and valid evidence-intervention probes.

## 1. Paradigm Shift

The original KnowSAM-style path is prediction-first:

```text
student prediction -> SAM prompt -> SAM pseudo label -> student distillation
```

RA-SAM-SSL changes the control flow:

```text
student hypothesis -> prompt family -> SAM response audit -> controllability attribution -> admitted pseudo-boundary learning
```

The student mask is treated as a hypothesis for querying SAM, not as truth. SAM is treated as a prompt-conditioned response instrument before it is allowed to become a teacher.

## 2. What Is Audited

For each unlabeled image, the implementation audits candidate SAM boundaries with:

1. Prompt-dose response:
   - Build SDF-shifted mask prompts from the student/fusion prediction.
   - Query SAM with `delta={-4,-2,0,2,4}` by default.
   - Estimate boundary response slope and jump instability.

2. Valid local evidence intervention:
   - Intervene only inside the zero-dose SAM boundary tube.
   - Use mild structure-preserving smoothing and local mean blending.
   - Check intervention locality and mildness before using ESR.

3. Evidence-sensitive response:
   - Re-query SAM on the intervened image.
   - Compute ESR as the response difference between original and intervened prompt-dose curves.

## 3. Boundary Attribution

The probe produces four response regimes:

| Regime | Operational condition | Training decision |
|---|---|---|
| Prompt-dominated | high prompt response slope | suppress strong boundary distillation |
| Prior-locked | low prompt slope and low ESR | suppress or defer |
| Evidence-sensitive | non-saturated, low/moderate prompt slope, high ESR, low jump | admit for boundary distillation |
| Unidentifiable | invalid, saturated, or not classified | weak consistency/interior KD only |

The key output is `admission_map`, not a claim of causal correctness. It only says the SAM boundary passed a response-controllability audit and can be used as stronger boundary supervision.

## 4. Code Mapping

```text
ra_modules.py
  ResponseAuditProbe
    build_prompt_family()
    run_curve()
    attenuate_image()
    intervention_validity()
    compute_admission()
    probe()

trainer_ra.py
  Trainer.train()
    student/fusion hypothesis
    SAM zero-dose teacher
    RA admission-gated boundary KD
    standard supervised and consistency losses

diagnose_ra.py
  validation-set response audit sanity check
  saves response curves, category heatmaps, boundary-error correlations

prediction_ra.py
  test-time SGDL/SAM evaluation and overlay export
```

## 5. Main Experimental Claims Supported by Code

The code directly supports these necessary experiments:

1. Main comparison:
   - Train full RA-SAM-SSL and compare Dice / IoU / HD95 against KnowSAM and SAM-guided variants.

2. Equal-query baseline:
   - `RA_BASELINE=prompt_ensemble` uses the same prompt family as an ensemble teacher.

3. Prompt/evidence ablation:
   - `RA_DISABLE_ESR=1`
   - `RA_DISABLE_PROMPT=1`
   - `RA_NO_INTERVENTION=1`

4. Intervention validity ablation:
   - `RA_INTERVENTION_MODE=boundary`
   - `RA_INTERVENTION_MODE=random`
   - `RA_INTERVENTION_MODE=interior`

5. Diagnostic mechanism analysis:
   - `diagnose_ra.py` reports admission, PC, ESR, jump, valid, and response category maps against labeled boundary error.

## 6. Important Writing Boundary

Do not describe this implementation as proving that a boundary is caused by image evidence. The defensible statement is:

```text
RA-SAM-SSL tests whether a SAM boundary response remains non-dominated by prompt control and is reproducibly sensitive to a validity-checked local evidence intervention.
```

This is a supervision admission principle, not merely prompt augmentation, confidence calibration, or a boundary-aware loss.
