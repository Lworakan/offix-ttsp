# Subject demographics (CIPCV 2026 / CVIPPR 2026 paper)

| ID | Name (internal) | Sex | Age | Height (cm) | Weight (kg) | BMI  | Split |
|----|-----------------|-----|-----|-------------|-------------|------|-------|
| S1 | pan    | M | 20–21 | 172 | 60 | 20.3 | test  |
| S2 | fu     | M | 20–21 | 170 | 50 | 17.3 | train |
| S3 | mukrop | M | 20–21 | 175 | 70 | 22.9 | train |
| S4 | nonny  | M | 20–21 | 176 | 63 | 20.3 | train |
| S5 | boom   | M | 21    | 180 | 81 | 25.0 | train |
| S6 | peemai | F | 20–21 | 165 | 47 | 17.3 | test  |
| S7 | mai    | F | 20–21 | 150 | 50 | 22.2 | train |
| S8 | money  | F | 20–21 | 170 | 60 | 20.8 | train |
| S9 | namoon | F | 20–21 | 155 | 65 | 27.1 | train |

- 9 subjects total: 5 male, 4 female
- Age range: 20–21, all university students from different universities
- Height range: 150–180 cm
- Weight range: 47–81 kg
- BMI range: 17.3–27.1 kg/m² (underweight to overweight)
- All 9 subjects participate in strict 9-fold leave-one-subject-out (LOSO) evaluation in the CVIPPR 2026 paper; the `split` column above only governs the legacy 6-subject train/test split used by earlier CIPCV experiments and is not used for the LOSO results.
