## Robustness Table 3 -- Failure-mode decomposition (challenging conditions)

| Condition | Architecture | Level | Gen | Safe-int. | Asset reach | Unsafe | Timeout | Collision |
|---|---|---|---|---|---|---|---|---|
| R1_measurement_noise:high | PPO-C | C0 | d0_plus | 0.547 | 0.433 | 0.020 | 0.000 | 0.000 |
| R1_measurement_noise:high | PPO-C | C0 | d1 | 0.593 | 0.403 | 0.003 | 0.000 | 0.000 |
| R1_measurement_noise:high | PPO-C | C2 | d0_plus | 0.373 | 0.573 | 0.010 | 0.000 | 0.020 |
| R1_measurement_noise:high | PPO-C | C2 | d1 | 0.370 | 0.543 | 0.010 | 0.000 | 0.050 |
| R1_measurement_noise:high | PPO-Kalman-C | C0 | d0_plus | 0.420 | 0.567 | 0.013 | 0.000 | 0.000 |
| R1_measurement_noise:high | PPO-Kalman-C | C0 | d1 | 0.640 | 0.350 | 0.010 | 0.000 | 0.000 |
| R1_measurement_noise:high | PPO-Kalman-C | C2 | d0_plus | 0.347 | 0.583 | 0.007 | 0.003 | 0.043 |
| R1_measurement_noise:high | PPO-Kalman-C | C2 | d1 | 0.433 | 0.503 | 0.013 | 0.003 | 0.033 |
| R1_measurement_noise:high | LR-PPO-C | C0 | d0_plus | 0.460 | 0.530 | 0.010 | 0.000 | 0.000 |
| R1_measurement_noise:high | LR-PPO-C | C0 | d1 | 0.400 | 0.590 | 0.010 | 0.000 | 0.000 |
| R1_measurement_noise:high | LR-PPO-C | C2 | d0_plus | 0.517 | 0.447 | 0.003 | 0.000 | 0.013 |
| R1_measurement_noise:high | LR-PPO-C | C2 | d1 | 0.477 | 0.473 | 0.003 | 0.000 | 0.033 |
| R2_detection_radius:short | PPO-C | C0 | d0_plus | 0.413 | 0.390 | 0.197 | 0.000 | 0.000 |
| R2_detection_radius:short | PPO-C | C0 | d1 | 0.337 | 0.477 | 0.187 | 0.000 | 0.000 |
| R2_detection_radius:short | PPO-C | C2 | d0_plus | 0.330 | 0.507 | 0.127 | 0.000 | 0.027 |
| R2_detection_radius:short | PPO-C | C2 | d1 | 0.293 | 0.577 | 0.093 | 0.003 | 0.017 |
| R2_detection_radius:short | PPO-Kalman-C | C0 | d0_plus | 0.333 | 0.480 | 0.187 | 0.000 | 0.000 |
| R2_detection_radius:short | PPO-Kalman-C | C0 | d1 | 0.447 | 0.353 | 0.200 | 0.000 | 0.000 |
| R2_detection_radius:short | PPO-Kalman-C | C2 | d0_plus | 0.273 | 0.560 | 0.137 | 0.003 | 0.017 |
| R2_detection_radius:short | PPO-Kalman-C | C2 | d1 | 0.287 | 0.540 | 0.127 | 0.003 | 0.017 |
| R2_detection_radius:short | LR-PPO-C | C0 | d0_plus | 0.340 | 0.437 | 0.223 | 0.000 | 0.000 |
| R2_detection_radius:short | LR-PPO-C | C0 | d1 | 0.353 | 0.430 | 0.217 | 0.000 | 0.000 |
| R2_detection_radius:short | LR-PPO-C | C2 | d0_plus | 0.493 | 0.373 | 0.127 | 0.000 | 0.000 |
| R2_detection_radius:short | LR-PPO-C | C2 | d1 | 0.460 | 0.400 | 0.120 | 0.000 | 0.007 |
| R3_relative_mobility:defender_disadvantaged | PPO-C | C0 | d0_plus | 0.670 | 0.293 | 0.037 | 0.000 | 0.000 |
| R3_relative_mobility:defender_disadvantaged | PPO-C | C0 | d1 | 0.453 | 0.543 | 0.003 | 0.000 | 0.000 |
| R3_relative_mobility:defender_disadvantaged | PPO-C | C2 | d0_plus | 0.513 | 0.443 | 0.017 | 0.000 | 0.013 |
| R3_relative_mobility:defender_disadvantaged | PPO-C | C2 | d1 | 0.413 | 0.540 | 0.010 | 0.000 | 0.013 |
| R3_relative_mobility:defender_disadvantaged | PPO-Kalman-C | C0 | d0_plus | 0.480 | 0.507 | 0.013 | 0.000 | 0.000 |
| R3_relative_mobility:defender_disadvantaged | PPO-Kalman-C | C0 | d1 | 0.653 | 0.320 | 0.027 | 0.000 | 0.000 |
| R3_relative_mobility:defender_disadvantaged | PPO-Kalman-C | C2 | d0_plus | 0.430 | 0.540 | 0.007 | 0.000 | 0.013 |
| R3_relative_mobility:defender_disadvantaged | PPO-Kalman-C | C2 | d1 | 0.573 | 0.383 | 0.010 | 0.000 | 0.027 |
| R3_relative_mobility:defender_disadvantaged | LR-PPO-C | C0 | d0_plus | 0.770 | 0.210 | 0.020 | 0.000 | 0.000 |
| R3_relative_mobility:defender_disadvantaged | LR-PPO-C | C0 | d1 | 0.780 | 0.200 | 0.020 | 0.000 | 0.000 |
| R3_relative_mobility:defender_disadvantaged | LR-PPO-C | C2 | d0_plus | 0.737 | 0.243 | 0.013 | 0.000 | 0.003 |
| R3_relative_mobility:defender_disadvantaged | LR-PPO-C | C2 | d1 | 0.720 | 0.237 | 0.020 | 0.000 | 0.007 |