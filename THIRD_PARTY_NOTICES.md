# External assets and project license status

This repository contains the project's optimization implementations and small result exports. It does not vendor the full Fengshenbang-LM checkout, the pretrained UniMC weights, or benchmark datasets.

- Official code: [IDEA-CCNL/Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM), whose upstream repository specifies [Apache License 2.0](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/LICENSE).
- Base model: [IDEA-CCNL/Erlangshen-UniMC-Albert-235M-English](https://huggingface.co/IDEA-CCNL/Erlangshen-UniMC-Albert-235M-English), revision `e3aeebc7bd475d559bd0cbd8d17abf4e159d1260`. Obtain and use it under its upstream asset terms.
- Source benchmarks: PIQA and CommonsenseQA, downloaded separately by the preparation script. Dataset contents are not included here.
- LoRA method: [Hu et al., LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
- UniMC paper: [Zero-Shot Learners for Natural Language Understanding via a Unified Multiple Choice Perspective](https://aclanthology.org/2022.emnlp-main.474/).

No new project-wide license has been assigned by this repository preparation. The project team should decide the license for its own contributions before an open-source release; this file does not change third-party rights or terms.
