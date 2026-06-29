-- ============================================
-- Used Vehicle Pricing Analysis
-- Dataset: Craigslist Used Cars (360k+ rows)
-- ============================================
-- --------------------------------------------
-- QUERY 1: Average Price by Manufacturer and Year
-- Business Question: How have prices trended 
-- across brands over model years?
-- --------------------------------------------
SELECT manufacturer,
    year,
    ROUND(AVG(price), 2) as avg_price,
    COUNT(*) as num_listings
FROM listings
WHERE price > 500
    AND price < 100000
    AND manufacturer IS NOT NULL
    AND year IS NOT NULL
GROUP BY manufacturer,
    year
ORDER BY manufacturer,
    year;
-- --------------------------------------------
-- QUERY 2: Price by Mileage Bucket
-- Business Question: How does odometer reading
-- affect resale value?
-- Finding: Clear depreciation curve — ~69% price
-- drop from under 30k to over 150k miles
-- --------------------------------------------
SELECT CASE
        WHEN odometer < 30000 THEN '1. Under 30k'
        WHEN odometer BETWEEN 30000 AND 60000 THEN '2. 30k-60k'
        WHEN odometer BETWEEN 60000 AND 100000 THEN '3. 60k-100k'
        WHEN odometer BETWEEN 100000 AND 150000 THEN '4. 100k-150k'
        ELSE '5. Over 150k'
    END as mileage_bucket,
    ROUND(AVG(price), 2) as avg_price,
    COUNT(*) as num_listings
FROM listings
WHERE price > 500
    AND price < 100000
    AND odometer IS NOT NULL
    AND year >= 2000
GROUP BY mileage_bucket
ORDER BY mileage_bucket;
-- --------------------------------------------
-- QUERY 3: Top 10 Manufacturers by Listing Volume
-- Business Question: Which brands dominate 
-- the used car supply?
-- --------------------------------------------
SELECT manufacturer,
    COUNT(*) as num_listings
FROM listings
WHERE manufacturer IS NOT NULL
GROUP BY manufacturer
ORDER BY num_listings DESC
LIMIT 10;
-- --------------------------------------------
-- QUERY 4: Manufacturers Priced Above Market Average
-- Business Question: Which brands command 
-- above-average resale prices?
-- Uses subquery to calculate market benchmark
-- --------------------------------------------
SELECT manufacturer,
    ROUND(AVG(price), 2) as avg_price
FROM listings
WHERE price > 500
    AND price < 100000
    AND manufacturer IS NOT NULL
GROUP BY manufacturer
HAVING avg_price > (
        SELECT ROUND(AVG(price), 2)
        FROM listings
        WHERE price > 500
            AND price < 100000
    )
ORDER BY avg_price DESC;
-- --------------------------------------------
-- QUERY 5: RAM Truck Breakdown by Vehicle Type
-- Business Question: Why does RAM outperform 
-- European luxury brands in average price?
-- Finding: Pickup trucks (6,868 listings at 
-- $31596 avg) dominate RAM's portfolio
-- --------------------------------------------
SELECT type,
    COUNT(*) as num_listings,
    ROUND(AVG(price), 2) as avg_price
FROM listings
WHERE price > 500
    AND price < 100000
    AND manufacturer LIKE 'ram'
    AND type IS NOT NULL
GROUP BY type
ORDER BY num_listings DESC;
-- --------------------------------------------
-- QUERY 6: Price by Country of Origin and Brand Tier
-- Business Question: Do domestic vs foreign 
-- and luxury vs mass-market segments show 
-- meaningful price differences?
-- Requires JOIN with manufacturer_origin table
-- --------------------------------------------
SELECT country,
    brand_tier,
    ROUND(AVG(price), 2) as avg_price,
    COUNT(*) as num_listings
FROM listings as l
    RIGHT JOIN manufacturer_origin as m ON l.manufacturer = m.manufacturer
WHERE (
        l.price > 500
        AND l.price < 100000
    )
    OR l.price IS NULL
GROUP BY country,
    brand_tier
ORDER BY avg_price DESC;