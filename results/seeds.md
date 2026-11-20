# Seed variation

Same data and settings, 3 training seeds (0, 1, 2). Mean ± sample standard deviation, in percent.

| Seed | Val acc | Test acc | Hard acc | Test macro-F1 (x100) | Hard macro-F1 (x100) |
|---|---|---|---|---|---|
| 0 | 93.6 | 89.4 | 70.9 | 88.7 | 70.8 |
| 1 | 92.7 | 90.2 | 74.5 | 89.8 | 74.3 |
| 2 | 94.5 | 90.2 | 70.9 | 90.0 | 69.4 |
| **mean ± sd** | 93.6 ± 0.9 | 89.9 ± 0.4 | 72.1 ± 2.1 | 89.5 ± 0.7 | 71.5 ± 2.5 |

Review policy (confidence >= 0.9 and >= 5 words), mean over seeds:

| Split | Auto-sorted | Accuracy of those |
|---|---|---|
| test | 85.9 ± 1.2% | 94.7 ± 1.6% |
| hard | 41.8 ± 3.6% | 82.8 ± 2.9% |
