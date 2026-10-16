## Robustness Table 1 -- Safe-interception rate (D0+ / D1)

| Architecture | Level | Condition | D0+ | D1 |
|---|---|---|---|---|
| PPO-C | C0 | nominal | 0.720 | 0.457 |
| PPO-C | C0 | measurement_noise: low | 0.690 | 0.430 |
| PPO-C | C0 | measurement_noise: high | 0.547 | 0.593 |
| PPO-C | C0 | detection_radius: short | 0.413 | 0.337 |
| PPO-C | C0 | detection_radius: long | 0.603 | 0.373 |
| PPO-C | C0 | relative_mobility: defender_disadvantaged | 0.670 | 0.453 |
| PPO-C | C0 | relative_mobility: defender_advantaged | 0.373 | 0.287 |
| PPO-C | C2 | nominal | 0.440 | 0.443 |
| PPO-C | C2 | measurement_noise: low | 0.453 | 0.357 |
| PPO-C | C2 | measurement_noise: high | 0.373 | 0.370 |
| PPO-C | C2 | detection_radius: short | 0.330 | 0.293 |
| PPO-C | C2 | detection_radius: long | 0.380 | 0.323 |
| PPO-C | C2 | relative_mobility: defender_disadvantaged | 0.513 | 0.413 |
| PPO-C | C2 | relative_mobility: defender_advantaged | 0.263 | 0.260 |
| PPO-Kalman-C | C0 | nominal | 0.527 | 0.713 |
| PPO-Kalman-C | C0 | measurement_noise: low | 0.530 | 0.720 |
| PPO-Kalman-C | C0 | measurement_noise: high | 0.420 | 0.640 |
| PPO-Kalman-C | C0 | detection_radius: short | 0.333 | 0.447 |
| PPO-Kalman-C | C0 | detection_radius: long | 0.427 | 0.613 |
| PPO-Kalman-C | C0 | relative_mobility: defender_disadvantaged | 0.480 | 0.653 |
| PPO-Kalman-C | C0 | relative_mobility: defender_advantaged | 0.237 | 0.473 |
| PPO-Kalman-C | C2 | nominal | 0.330 | 0.497 |
| PPO-Kalman-C | C2 | measurement_noise: low | 0.377 | 0.480 |
| PPO-Kalman-C | C2 | measurement_noise: high | 0.347 | 0.433 |
| PPO-Kalman-C | C2 | detection_radius: short | 0.273 | 0.287 |
| PPO-Kalman-C | C2 | detection_radius: long | 0.343 | 0.393 |
| PPO-Kalman-C | C2 | relative_mobility: defender_disadvantaged | 0.430 | 0.573 |
| PPO-Kalman-C | C2 | relative_mobility: defender_advantaged | 0.217 | 0.390 |
| LR-PPO-C | C0 | nominal | 0.700 | 0.573 |
| LR-PPO-C | C0 | measurement_noise: low | 0.800 | 0.750 |
| LR-PPO-C | C0 | measurement_noise: high | 0.460 | 0.400 |
| LR-PPO-C | C0 | detection_radius: short | 0.340 | 0.353 |
| LR-PPO-C | C0 | detection_radius: long | 0.607 | 0.570 |
| LR-PPO-C | C0 | relative_mobility: defender_disadvantaged | 0.770 | 0.780 |
| LR-PPO-C | C0 | relative_mobility: defender_advantaged | 0.687 | 0.593 |
| LR-PPO-C | C2 | nominal | 0.740 | 0.647 |
| LR-PPO-C | C2 | measurement_noise: low | 0.733 | 0.693 |
| LR-PPO-C | C2 | measurement_noise: high | 0.517 | 0.477 |
| LR-PPO-C | C2 | detection_radius: short | 0.493 | 0.460 |
| LR-PPO-C | C2 | detection_radius: long | 0.650 | 0.650 |
| LR-PPO-C | C2 | relative_mobility: defender_disadvantaged | 0.737 | 0.720 |
| LR-PPO-C | C2 | relative_mobility: defender_advantaged | 0.637 | 0.603 |