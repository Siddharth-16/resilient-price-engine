# SQL & Tableau Pricing Analysis

Supplementary analysis of the Craigslist vehicle dataset using
MySQL and Tableau, focusing on pricing trends, market segmentation,
and depreciation patterns.

## Business Questions Answered

1. How do average prices trend across manufacturers and model years?
2. How does mileage affect resale value?
3. Which manufacturers dominate used car supply?
4. Which brands price above the overall market average of $19,202?
5. What drives RAM trucks to outperform European luxury brands?
6. How do prices differ by country of origin and brand tier?

## Key Findings

- **70% price depreciation** from under 30k miles to over
  150k miles
- **RAM ($31,056) outprice Mercedes-Benz ($23,177)** — driven
  by high-volume pickup listings averaging $32,101
- **Ford dominates supply** with 58,856 listings, nearly 2x Toyota
- **20 manufacturers price above market average of $19,603**, led by
  Ferrari, Aston-Martin, and Tesla

## Dashboard

Interactive Tableau dashboard built on 348k+ filtered listings:

![Dashboard](Dashboard.png)

## Files

- `queries.sql` — All analysis queries with comments explaining
  business context
- `dashboard.png` — Interactive Tableau market overview dashboard
