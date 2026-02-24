# Data

Place `Employee_Data.csv` in this directory before running the pipeline.

## Expected Schema

| Column             | Type          | Description                                         |
|--------------------|---------------|-----------------------------------------------------|
| Name               | string        | Anonymized employee name                            |
| Salary             | float         | Annual base salary                                  |
| DOJ                | date          | Date of joining                                     |
| Age                | int           | Current age                                         |
| Gender             | string        | Male, Female, Other                                 |
| Dependents         | int           | Number of dependents                                |
| HA                 | float         | Housing Allowance component                         |
| COLA               | float         | Cost of Living Adjustment component                 |
| CPF                | float         | Central Provident Fund contribution                 |
| Gross Salary       | float         | Total annual compensation (all components)          |
| Insurance          | string        | None, Life, Health, Both                            |
| Marital Status     | string        | Single, Married, Divorced, Widowed                  |
| In Company Years   | int           | Tenure at the company                               |
| Year of Experience | int           | Total professional experience                       |
| Department         | string        | Sales, Marketing, Finance, HR, etc.                 |
| Position           | string        | Granular job title (normalized during preprocessing)|

The CSV is not included in version control. It contains 4,600 employee records
from a multinational company across multiple departments.
