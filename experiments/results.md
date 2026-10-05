# Knowledge-retention ablation

| run | params | steps | tokens | tok/s | fineweb val ↓ | wiki seen ↓ | wiki unseen ↓ | facts acc ↑ | facts logprob ↑ |
|---|---|---|---|---|---|---|---|---|---|
| e0_baseline | 179M | 5394 | 354M | 53555 | 3.744 | 3.990 | 4.018 | 17% | -4.10 |
| e1_datamix | 179M | 5387 | 353M | 53486 | 3.847 | 0.879 | 5.057 | 20% | -4.39 |
| e2_ple | 268M | 5090 | 334M | 50538 | 3.852 | 0.861 | 5.133 | 23% | -4.50 |
| checkpoints | 179M | 3000 | 0M |  | 3.861 | 4.096 | 4.117 | 20% | -4.41 |
| checkpoints2 | 189M | 1500 | 0M |  | 3.982 | 4.254 | 4.274 | 13% | -5.93 |
