# SRG-SAM / SRG-SAM++-Lite

SRG-SAM (Structure-Reliability Guided SAM) is a semi-supervised ultrasound
segmentation variant derived from the multiclass KnowSAM code path.

Core SRG-SAM rule:

```text
SAM is frozen and is used only inside SSRF to estimate structural reliability.
SAM masks are not pseudo labels and are not used for semantic supervision.
```

SRG-SAM++-Lite uses SAM during training, but removes SAM from deployment:

```text
Frozen-SAM structure branch:
  training only. Estimates R_b/R_a/R_u as structural reliability, not labels.

Adapter-SAM semantic branch:
  training only. Keeps KnowSAM Adapter/super_prompt trainability and provides
  soft semantic auxiliary logits, but never hard pseudo labels.

DPG graph logits branch:
  training and inference. Provides class-relation logits from dynamic prototypes.

Reliability Head:
  training and inference. Learns R_hat from student features to replace SAM SSRF.

Deploy branch:
  deploy_logits = student_logits + alpha_graph * R_hat * graph_logits
```

The main validation/test result is `deploy_multiclass_val`, not the optional
SAM-assisted upper bound.

Implemented modules:

```text
models/ssrf.py          SSRF: R_b, R_a, R_u from multi-prompt SAM stability
losses/srpc_loss.py     SRPC: R_b * (1 - R_u) weighted pseudo posterior loss
models/dynamic_graph.py DPG: dynamic prototype graph with R_prior/R_sam blend
train_ssl.py            SRG-SAM training entry
models/reliability_head.py Lite deploy-time R_hat head
trainer.py              SRG-SAM++-Lite training loop and no-SAM validation
```

SRG-SAM++-Lite training objective:

```text
loss_total = loss_student_sup
           + lambda_deploy * loss_deploy_sup
           + lambda_sam * loss_sam_sup
           + lambda_graph * loss_graph
           + lambda_R * loss_R
           + lambda_u * loss_u
           + lambda_struct * loss_struct
           + lambda_graph_reg * loss_graph_reg
```

Run:

```bash
bash ./variants/Multiclass_KnowSAM/train_v100_multiclass.sh
```

Main ablation switches:

```bash
--use_dpg_deploy false --use_reliability_head false  # student only
--use_dpg_deploy true --use_reliability_head false    # student + DPG, no R_hat
--use_dpg_deploy true --use_reliability_head true     # deploy branch
--lambda_sam 0 --beta_sam 0                           # remove Adapter-SAM distillation
--lambda_R 0 --lambda_struct 0                        # remove Frozen-SAM reliability distillation
--eval_full_sam_assisted true                         # optional upper bound, not main result
```

Default V100 labeled30pct fine-tune:

```bash
DATASET=/260513_data_labeled30pct NUM_CLASSES=3 \
bash ./variants/Multiclass_KnowSAM/finetune_v100_labeled30pct.sh
```

Default no-SAM test:

```bash
DATASET=/260513_data_labeled30pct NUM_CLASSES=3 \
SNAPSHOT_PATH=./Results/SRG-SAM_PP_V100_260513_data_labeled30pct_from_scratch \
bash ./variants/Multiclass_KnowSAM/test_v100_multiclass.sh
```

For binary BUSI-style segmentation:

```bash
DATA_PATH=/path/to/root DATASET=/BUSI_semi NUM_CLASSES=2 \
bash ./variants/Multiclass_KnowSAM/train_v100_multiclass.sh
```

Expected dataset layout:

```text
root/BUSI_semi/
  labeled/image
  labeled/mask
  unlabeled/image
  val/image
  val/mask
```

## Complementarity-aware SRG-SAM++ v2

The new path is opt-in and keeps the legacy SRG-SAM++-Lite defaults unchanged.
Enable it with:

```text
--use_ca_srg_sampp true
--graph_type region_boundary_prototype
--reliability_type complementarity
--reliability_target_type delta_utility
--deploy_fusion_type gated_residual
```

Main behavior:

```text
Stage student:
  deploy_logits == student_logits
  train SGDL student with EMA weak-strong SSL and view consistency.

Stage residual / joint:
  residual_logits = RegionBoundaryPrototypeGraph(...)
  R_delta_hat = ComplementarityReliabilityHead(...)
  deploy_logits = student_logits + lambda_res * R_delta_hat * residual_logits
```

The graph is no longer treated as a full segmentation branch in the CA path. It
only predicts residual correction logits. Reliability is trained against
patch-level correction utility relative to the student instead of frozen-SAM
structural reliability.

Example staged runs:

```bash
# Stage 1: student only
TRAIN_STAGE=student \
SNAPSHOT_PATH=./Results/CA_SRG_SAMPP_labeled30pct_stage1 \
bash ./variants/Multiclass_KnowSAM/train_ca_srg_sampp_labeled30pct.sh

# Stage 2: residual graph and complementarity reliability
TRAIN_STAGE=residual \
SGDL_INIT_CHECKPOINT=./Results/CA_SRG_SAMPP_labeled30pct_stage1/best_student_model.pth \
SNAPSHOT_PATH=./Results/CA_SRG_SAMPP_labeled30pct_stage2 \
bash ./variants/Multiclass_KnowSAM/train_ca_srg_sampp_labeled30pct.sh

# Stage 3: joint fine-tuning
TRAIN_STAGE=joint \
SGDL_INIT_CHECKPOINT=./Results/CA_SRG_SAMPP_labeled30pct_stage2/best_deploy_model.pth \
SNAPSHOT_PATH=./Results/CA_SRG_SAMPP_labeled30pct_stage3 \
bash ./variants/Multiclass_KnowSAM/train_ca_srg_sampp_labeled30pct.sh
```

To restore the original SRG-SAM++-Lite path, omit `--use_ca_srg_sampp true` or
set:

```bash
--use_ca_srg_sampp false --graph_type legacy --reliability_type old_R --deploy_fusion_type weighted_sum
```

New validation diagnostics include `student_multiclass_val`,
`deploy_multiclass_val`, `corrected_multiclass_val`,
`oracle_fusion_avg_dice`, `oracle_gain_over_student`,
`R_delta_improvement_corr`, `A_diag_mean`, `A_offdiag_mean`, `A_entropy`, and
`deploy_student_delta`. Oracle fusion uses GT only for diagnostics and never to
form deploy logits.
