# SQL Pricing Analysis

Supplementary analysis of the Craigslist vehicle dataset using MySQL,
focusing on pricing trends, market segmentation, and depreciation patterns.

## Business Questions Answered

1. How do average prices trend across manufacturers and model years?
2. How does mileage affect resale value?
3. Which manufacturers dominate used car supply?
4. Which brands price above the overall market average of $19,202?
5. What drives RAM trucks to outperform European luxury brands in avg price?
6. How do prices differ by country of origin and brand tier?

## Key Findings

- **~69% price depreciation** from under 30k miles ($31,551) to over
  150k miles ($9,899)
- **RAM trucks ($30,290) outprice Mercedes-Benz ($22,217)** — driven by
  high-volume pickup listings averaging $31,596
- **Ford dominates supply** with 70,985 listings, nearly 2x Toyota
- **20 manufacturers price above market average**, led by Ferrari,
  Aston-Martin, and Tesla
- **Japanese mass-market brands most affordable** at $14,782 avg vs
  USA mass-market at $21,764

## Files

- `queries.sql` — All analysis queries with comments explaining
  business context
