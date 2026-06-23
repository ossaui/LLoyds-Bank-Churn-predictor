# Data Notes

## Included Data Files

- `Advanced_Churn_Analysis_Dataset.csv`
- `Customer_Churn_Data_Large.xlsx`

## Modeling Target

- Target column: `ChurnStatus`
- Problem type: binary classification
- Positive class: churned customer

## Dataset Summary

- Rows: 1,000
- Columns: 27
- Churn customers: 204
- Non-churn customers: 796
- Churn rate: 20.4%

## Columns

- `CustomerID`
- `ChurnStatus`
- `LoginFrequency`
- `AvgAmountSpent`
- `SpendingVolatility`
- `MaxAmountSpent`
- `IsMarried`
- `IsDivorced`
- `IncomeLow`
- `IncomeHigh`
- `Age`
- `IsMale`
- `TotalTransactions`
- `TotalAmountSpent`
- `CategoryDiversity`
- `DaysSinceLastTransaction`
- `CustomerTenureDays`
- `ComplaintCount`
- `ComplaintRatio`
- `HasUnresolvedComplaint`
- `ServiceResolutionRate`
- `TotalServiceInteractions`
- `DaysSinceLastLogin`
- `UsesMobileApp`
- `UsesWebsite`
- `PrefersElectronics`
- `PrefersClothing`

## Modeling Treatment

`CustomerID` is treated as an identifier and removed from model training. The remaining columns are numeric or already encoded indicator variables.
