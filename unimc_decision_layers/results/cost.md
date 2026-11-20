| model | params | latency ms / 1000 samples | peak GPU MB |
|---|---|---|---|
| UniMC baseline (argmax of raw logits) | 0 | 0 | 0 |
| mlp | 1,387,521 | 2.9 | 73 |
| tf_noint | 1,651,201 | 4.1 | 74 |
| tf_int | 1,651,201 | 3.3 | 74 |
| tf_gate | 1,651,714 | 3.6 | 74 |
