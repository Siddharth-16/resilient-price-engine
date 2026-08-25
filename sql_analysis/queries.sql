-- ============================================================
-- ANALYSIS 1: PRICE BY AGE AND MILEAGE
-- ============================================================
-- Question:
-- How does mileage relate to vehicle price within similar age groups?
-- Includes mean, median, and sample size.
-- ============================================================
WITH bucketed AS (
    SELECT price,
        CASE
            WHEN car_age <= 3 THEN '1. 0-3 years'
            WHEN car_age <= 7 THEN '2. 4-7 years'
            WHEN car_age <= 12 THEN '3. 8-12 years'
            ELSE '4. 13+ years'
        END AS age_bucket,
        CASE
            WHEN odometer < 30000 THEN '1. Under 30k'
            WHEN odometer < 60000 THEN '2. 30k-60k'
            WHEN odometer < 100000 THEN '3. 60k-100k'
            WHEN odometer < 150000 THEN '4. 100k-150k'
            ELSE '5. Over 150k'
        END AS mileage_bucket
    FROM listings
),
ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY age_bucket,
            mileage_bucket
            ORDER BY price
        ) AS rn,
        COUNT(*) OVER (
            PARTITION BY age_bucket,
            mileage_bucket
        ) AS cnt
    FROM bucketed
)
SELECT age_bucket,
    mileage_bucket,
    COUNT(*) AS num_listings,
    ROUND(AVG(price), 2) AS mean_price,
    ROUND(
        AVG(
            CASE
                WHEN rn IN (
                    FLOOR((cnt + 1) / 2),
                    FLOOR((cnt + 2) / 2)
                ) THEN price
            END
        ),
        2
    ) AS median_price
FROM ranked
GROUP BY age_bucket,
    mileage_bucket
HAVING COUNT(*) >= 200
ORDER BY age_bucket,
    mileage_bucket;
-- ============================================================
-- ANALYSIS 1A: INVESTIGATE 4-7 YEAR / 150K+ MILEAGE ANOMALY
-- ============================================================
-- Question:
-- Is the unusually high price in this bucket explained
-- by vehicle-type composition?
-- ============================================================
SELECT type,
    COUNT(*) AS num_listings,
    ROUND(AVG(price), 2) AS mean_price
FROM listings
WHERE car_age BETWEEN 4 AND 7
    AND odometer >= 150000
GROUP BY type
HAVING COUNT(*) >= 50
ORDER BY num_listings DESC;
-- ============================================================
-- ANALYSIS 1B: SAME ANOMALY BY MANUFACTURER
-- ============================================================
SELECT manufacturer,
    COUNT(*) AS num_listings,
    ROUND(AVG(price), 2) AS mean_price
FROM listings
WHERE car_age BETWEEN 4 AND 7
    AND odometer >= 150000
GROUP BY manufacturer
HAVING COUNT(*) >= 50
ORDER BY num_listings DESC;
-- ============================================================
-- ANALYSIS 2: MANUFACTURER × VEHICLE TYPE COMPOSITION
-- ============================================================
-- Question:
-- How do vehicle-type differences affect manufacturer-level
-- pricing comparisons?
-- Includes mean, median, and sample-size control.
-- ============================================================
WITH ranked_segments AS (
    SELECT manufacturer,
        type,
        price,
        ROW_NUMBER() OVER (
            PARTITION BY manufacturer,
            type
            ORDER BY price
        ) AS rn,
        COUNT(*) OVER (
            PARTITION BY manufacturer,
            type
        ) AS cnt
    FROM listings
    WHERE manufacturer <> 'unknown'
        AND type <> 'unknown'
)
SELECT manufacturer,
    type,
    COUNT(*) AS num_listings,
    ROUND(AVG(price), 2) AS mean_price,
    ROUND(
        AVG(
            CASE
                WHEN rn IN (
                    FLOOR((cnt + 1) / 2),
                    FLOOR((cnt + 2) / 2)
                ) THEN price
            END
        ),
        2
    ) AS median_price
FROM ranked_segments
GROUP BY manufacturer,
    type
HAVING COUNT(*) >= 200
ORDER BY manufacturer,
    num_listings DESC;
-- ============================================================
-- ANALYSIS 3: MODEL PREDICTION ERROR DIAGNOSTICS
-- ============================================================
-- Objective:
-- Evaluate whether the price prediction model performs
-- consistently across different vehicle segments.
--
-- Overall MAE alone can hide poor performance for specific
-- manufacturers, vehicle ages, or mileage groups.
--
-- Residual definition:
--
--     residual = actual_price - predicted_price
--
-- Therefore:
--
--     positive residual -> model underpredicts
--     negative residual -> model overpredicts
--
-- absolute_error measures prediction magnitude regardless
-- of direction.
--
-- Minimum sample thresholds are used to avoid drawing
-- conclusions from very small groups.
-- ============================================================
-- ------------------------------------------------------------
-- ANALYSIS 3A: MODEL ERROR BY MANUFACTURER
-- ------------------------------------------------------------
-- Business Question:
-- Which manufacturers are hardest for the model to price
-- accurately?
--
-- Metrics:
--   MAE          -> average absolute prediction error
--   avg_residual -> average direction of model bias
--
-- Interpretation:
--   avg_residual > 0 -> model tends to underpredict
--   avg_residual < 0 -> model tends to overpredict
--
-- Only manufacturers with at least 200 held-out predictions
-- are included.
-- ------------------------------------------------------------
SELECT manufacturer,
    COUNT(*) AS num_predictions,
    ROUND(AVG(price), 2) AS avg_actual_price,
    ROUND(AVG(absolute_error), 2) AS mae,
    ROUND(AVG(residual), 2) AS avg_residual,
    ROUND(
        SUM(absolute_error) / SUM(price) * 100,
        2
    ) AS wape
FROM model_predictions
WHERE manufacturer <> 'unknown'
GROUP BY manufacturer
HAVING COUNT(*) >= 200
ORDER BY mae DESC;
-- ------------------------------------------------------------
-- ANALYSIS 3B: MODEL ERROR BY VEHICLE AGE
-- ------------------------------------------------------------
-- Business Question:
-- Does model accuracy change as vehicles get older?
--
-- Vehicles are divided into the same age buckets used in the
-- market analysis so model performance can be compared against
-- previously observed pricing patterns.
--
-- This helps identify whether older or newer vehicles are
-- systematically more difficult to predict.
-- ------------------------------------------------------------
SELECT CASE
        WHEN car_age <= 3 THEN '1. 0-3 years'
        WHEN car_age <= 7 THEN '2. 4-7 years'
        WHEN car_age <= 12 THEN '3. 8-12 years'
        ELSE '4. 13+ years'
    END AS age_bucket,
    COUNT(*) AS num_predictions,
    ROUND(AVG(price), 2) AS avg_actual_price,
    ROUND(AVG(absolute_error), 2) AS mae,
    ROUND(AVG(residual), 2) AS avg_residual,
    ROUND(
        SUM(absolute_error) / SUM(price) * 100,
        2
    ) AS wape
FROM model_predictions
GROUP BY age_bucket
HAVING COUNT(*) >= 200
ORDER BY age_bucket;
-- ------------------------------------------------------------
-- ANALYSIS 3C: MODEL ERROR BY MILEAGE
-- ------------------------------------------------------------
-- Business Question:
-- Does prediction accuracy change for vehicles with higher
-- mileage?
--
-- Uses the same mileage buckets as Analysis 1 so pricing
-- behavior and prediction performance can be compared directly.
-- ------------------------------------------------------------
SELECT CASE
        WHEN odometer < 30000 THEN '1. Under 30k'
        WHEN odometer < 60000 THEN '2. 30k-60k'
        WHEN odometer < 100000 THEN '3. 60k-100k'
        WHEN odometer < 150000 THEN '4. 100k-150k'
        ELSE '5. Over 150k'
    END AS mileage_bucket,
    COUNT(*) AS num_predictions,
    ROUND(AVG(price), 2) AS avg_actual_price,
    ROUND(AVG(absolute_error), 2) AS mae,
    ROUND(AVG(residual), 2) AS avg_residual,
    ROUND(
        SUM(absolute_error) / SUM(price) * 100,
        2
    ) AS wape
FROM model_predictions
GROUP BY mileage_bucket
HAVING COUNT(*) >= 200
ORDER BY mileage_bucket;
-- ------------------------------------------------------------
-- ANALYSIS 3D: MODEL ERROR BY VEHICLE TYPE
-- ------------------------------------------------------------
-- Business Question:
-- Are certain vehicle types consistently harder to price?
--
-- This connects directly to Analysis 2, where vehicle type
-- was shown to strongly influence price distributions.
--
-- If trucks, pickups, luxury-oriented segments, etc. have
-- substantially different MAE, aggregate model performance
-- may hide important segment-level weaknesses.
-- ------------------------------------------------------------
SELECT type,
    COUNT(*) AS num_predictions,
    ROUND(AVG(price), 2) AS avg_actual_price,
    ROUND(AVG(absolute_error), 2) AS mae,
    ROUND(AVG(residual), 2) AS avg_residual,
    ROUND(
        SUM(absolute_error) / SUM(price) * 100,
        2
    ) AS wape
FROM model_predictions
WHERE type <> 'unknown'
GROUP BY type
HAVING COUNT(*) >= 200
ORDER BY mae DESC;
-- ============================================================
-- ANALYSIS 3E: WHY ARE OLD, LOW-MILEAGE VEHICLES HARDER
-- TO PREDICT?
-- ============================================================
-- Background:
-- Analysis 3C showed substantially higher prediction error
-- for low-mileage vehicles in the baseline evaluation cohort:
--
--   Under 30k miles : MAE ≈ $3,457
--   30k-60k miles  : MAE ≈ $3,509
--   Over 150k miles: MAE ≈ $1,810
--
-- Because the evaluation cohort consists primarily of older
-- vehicles, unusually low mileage may identify a very different
-- vehicle population.
--
-- Goal:
-- Determine whether the higher error is explained by:
--
--   1. vehicle-type composition,
--   2. manufacturer composition,
--   3. higher price dispersion,
--   4. or whether low-mileage vehicles remain harder to predict
--      even within comparable vehicle segments.
-- ============================================================
-- ------------------------------------------------------------
-- ANALYSIS 3E-1: LOW-MILEAGE ERROR BY VEHICLE TYPE
-- ------------------------------------------------------------
-- Question:
-- Which vehicle types contribute most to prediction error
-- among old vehicles with fewer than 60k miles?
--
-- WAPE:
--   total absolute error / total actual price
--
-- Price standard deviation helps determine whether a segment
-- contains a particularly wide range of vehicle prices.
-- ------------------------------------------------------------
SELECT type,
    COUNT(*) AS num_predictions,
    ROUND(AVG(price), 2) AS avg_actual_price,
    ROUND(STDDEV_SAMP(price), 2) AS price_stddev,
    ROUND(AVG(absolute_error), 2) AS mae,
    ROUND(AVG(residual), 2) AS avg_residual,
    ROUND(
        SUM(absolute_error) / SUM(price) * 100,
        2
    ) AS wape
FROM model_predictions
WHERE car_age >= 8
    AND odometer < 60000
    AND type <> 'unknown'
GROUP BY type
HAVING COUNT(*) >= 50
ORDER BY mae DESC;
-- ------------------------------------------------------------
-- ANALYSIS 3E-2: LOW-MILEAGE ERROR BY MANUFACTURER
-- ------------------------------------------------------------
-- Question:
-- Are particular manufacturers responsible for unusually
-- high prediction error among old, low-mileage vehicles?
--
-- If expensive or heterogeneous manufacturers dominate the
-- high-error groups, manufacturer composition may explain
-- part of the mileage effect.
-- ------------------------------------------------------------
SELECT manufacturer,
    COUNT(*) AS num_predictions,
    ROUND(AVG(price), 2) AS avg_actual_price,
    ROUND(STDDEV_SAMP(price), 2) AS price_stddev,
    ROUND(AVG(absolute_error), 2) AS mae,
    ROUND(AVG(residual), 2) AS avg_residual,
    ROUND(
        SUM(absolute_error) / SUM(price) * 100,
        2
    ) AS wape
FROM model_predictions
WHERE car_age >= 8
    AND odometer < 60000
    AND manufacturer <> 'unknown'
GROUP BY manufacturer
HAVING COUNT(*) >= 50
ORDER BY mae DESC;
-- ------------------------------------------------------------
-- ANALYSIS 3E-3: CONTROL FOR VEHICLE TYPE
-- ------------------------------------------------------------
-- Question:
-- Does the low-mileage error remain after comparing vehicles
-- within the same vehicle type?
--
-- This is the most important diagnostic.
--
-- If trucks, SUVs, sedans, etc. each show higher MAE at low
-- mileage, then mileage is associated with higher uncertainty
-- even after controlling broadly for vehicle type.
--
-- If the MAE gap largely disappears within each type, then the
-- original mileage result was primarily a composition effect.
-- ------------------------------------------------------------
WITH mileage_segment AS (
    SELECT type,
        price,
        absolute_error,
        residual,
        CASE
            WHEN odometer < 30000 THEN '1. Under 30k'
            WHEN odometer < 60000 THEN '2. 30k-60k'
            WHEN odometer < 100000 THEN '3. 60k-100k'
            WHEN odometer < 150000 THEN '4. 100k-150k'
            ELSE '5. Over 150k'
        END AS mileage_bucket
    FROM model_predictions
    WHERE car_age >= 8
        AND type <> 'unknown'
)
SELECT type,
    mileage_bucket,
    COUNT(*) AS num_predictions,
    ROUND(AVG(price), 2) AS avg_actual_price,
    ROUND(STDDEV_SAMP(price), 2) AS price_stddev,
    ROUND(AVG(absolute_error), 2) AS mae,
    ROUND(AVG(residual), 2) AS avg_residual,
    ROUND(
        SUM(absolute_error) / SUM(price) * 100,
        2
    ) AS wape
FROM mileage_segment
GROUP BY type,
    mileage_bucket
HAVING COUNT(*) >= 100
ORDER BY type,
    mileage_bucket;
-- ============================================================
-- MARKET OVERVIEW: TOP MANUFACTURERS BY LISTING VOLUME
-- ============================================================
-- Context:
-- Shows the largest manufacturers represented in the cleaned
-- Craigslist vehicle population. This is descriptive context
-- rather than a primary analytical finding.
-- ============================================================
SELECT manufacturer,
    COUNT(*) AS num_listings
FROM listings
WHERE manufacturer <> 'unknown'
GROUP BY manufacturer
ORDER BY num_listings DESC
LIMIT 10;