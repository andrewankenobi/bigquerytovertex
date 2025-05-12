-- =================================================================================================
-- SQL SCRIPT FOR GOOGLE ANALYTICS REVENUE ANALYSIS AND FORECASTING WITH ARIMA & TimesFM
-- =================================================================================================
-- BEFORE RUNNING: Replace placeholders or set BQ scripting variables.
-- Set your project ID and dataset ID using the bq CLI:
-- bq query --use_legacy_sql=false --parameter='project_id:STRING:your-gcp-project-id' --parameter='dataset_id:STRING:your_bq_dataset_id' < arima_forecasting.sql
-- Or declare them in the script if using the BigQuery Console:
-- DECLARE project_id STRING DEFAULT 'your-gcp-project-id';
-- DECLARE dataset_id STRING DEFAULT 'your_bq_dataset_id';
-- =================================================================================================
-- This script performs the following operations using the public Google Analytics Sample dataset:
-- 1. Shows sample data from the GA sessions table.
-- 2. Creates an aggregated daily revenue dataset suitable for time series modeling.
-- 3. Creates an ARIMA_PLUS model using BigQuery ML to forecast total daily revenue.
-- 4. Generates forecasts using the ARIMA_PLUS model.
-- 5. Explains the ARIMA_PLUS forecast results.
-- 6. Generates forecasts using the TimesFM 2.0 model via AI.FORECAST.
-- 7. Creates a table combining actuals and TimesFM forecasts for plotting.
-- 8. Creates and queries a Contribution Analysis model for GA revenue drivers.
-- 9. Shows how to register the ARIMA model to Vertex AI Model Registry.
-- =================================================================================================

-- =================================================================================================
-- SECTION 1: EXPLORE BASE DATA (Google Analytics Sample)
-- =================================================================================================
-- Purpose: To view a sample of the raw Google Analytics data.
-- Source: bigquery-public-data.google_analytics_sample.ga_sessions_*
-- =================================================================================================

SELECT
  date,
  fullVisitorId,
  visitId,
  channelGrouping,
  totals.visits,
  totals.hits,
  totals.pageviews,
  totals.timeOnSite,
  totals.transactions,
  totals.transactionRevenue,
  device.deviceCategory,
  geoNetwork.country
FROM
  `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170731' -- Limiting scan for preview
LIMIT 100;

-- =================================================================================================
-- SECTION 2: PREPARE TIME SERIES DATA FOR ARIMA (Daily Revenue)
-- =================================================================================================
-- Purpose: To create a dataset aggregated by date, suitable for univariate ARIMA forecasting.
-- We aggregate total transaction revenue across all sessions for each day.
-- Note: We filter for dates where revenue is not NULL and aggregate over a year for a reasonable series.
-- =================================================================================================

CREATE OR REPLACE VIEW `${project_id}.${dataset_id}.ga_daily_revenue` AS
SELECT
  PARSE_DATE('%Y%m%d', date) AS transaction_date,
  SUM(totals.transactionRevenue)/1000000 AS total_daily_revenue -- Revenue is stored as micro-dollars
FROM
  `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE
  _TABLE_SUFFIX BETWEEN '20160801' AND '20170731' -- Use one full year of data
  AND totals.transactionRevenue IS NOT NULL
GROUP BY
  transaction_date;

-- Show sample of the aggregated daily revenue data
SELECT
  *
FROM
  `${project_id}.${dataset_id}.ga_daily_revenue`
ORDER BY
  transaction_date
LIMIT 100;

-- =================================================================================================
-- SECTION 3: CREATE ARIMA_PLUS MODEL (Daily Revenue Forecast)
-- =================================================================================================
-- Purpose: To create a time series forecasting model for daily revenue using BQML's ARIMA_PLUS.
-- We use the 'ga_daily_revenue' view created previously.
-- =================================================================================================

CREATE OR REPLACE MODEL `${project_id}.${dataset_id}.arima_ga_daily_revenue_model`
OPTIONS(
  MODEL_TYPE='ARIMA_PLUS',
  TIME_SERIES_TIMESTAMP_COL='transaction_date',
  TIME_SERIES_DATA_COL='total_daily_revenue',
  AUTO_ARIMA = TRUE,
  DATA_FREQUENCY = 'DAILY', -- Specify frequency for better seasonality detection
  HOLIDAY_REGION='US' -- Include US holidays effect
) AS
SELECT
  transaction_date,
  total_daily_revenue
FROM
  `${project_id}.${dataset_id}.ga_daily_revenue`;

-- =================================================================================================
-- SECTION 3.1: EVALUATE ARIMA_PLUS MODEL
-- =================================================================================================
-- Purpose: To evaluate the performance metrics of the trained ARIMA_PLUS model.
-- =================================================================================================
CREATE OR REPLACE TABLE `${project_id}.${dataset_id}.ga_arima_evaluation` AS
SELECT
  *
FROM
  ML.EVALUATE(MODEL `${project_id}.${dataset_id}.arima_ga_daily_revenue_model`);

-- =================================================================================================
-- SECTION 4: FORECAST WITH ARIMA_PLUS MODEL & STORE RESULTS
-- =================================================================================================
-- Purpose: To generate future daily revenue forecasts and store them in a table.
-- =================================================================================================
CREATE OR REPLACE TABLE `${project_id}.${dataset_id}.ga_arima_forecast` AS
SELECT
  *
FROM
  ML.FORECAST(MODEL `${project_id}.${dataset_id}.arima_ga_daily_revenue_model`,
              STRUCT(120 AS horizon, 0.95 AS confidence_level)); -- Forecast 120 days ahead

-- =================================================================================================
-- SECTION 4.1: DETECT ANOMALIES WITH ARIMA_PLUS MODEL
-- =================================================================================================
-- Purpose: To identify anomalies in the historical data using the trained ARIMA_PLUS model.
-- =================================================================================================
CREATE OR REPLACE TABLE `${project_id}.${dataset_id}.ga_arima_anomalies` AS
SELECT
  *
FROM
  ML.DETECT_ANOMALIES(MODEL `${project_id}.${dataset_id}.arima_ga_daily_revenue_model`,
                      STRUCT(0.95 AS anomaly_prob_threshold)); -- Detect points with > 95% anomaly probability

-- =================================================================================================
-- SECTION 5: EXPLAIN ARIMA_PLUS FORECAST
-- =================================================================================================
-- Purpose: To understand the components contributing to the daily revenue forecast.
-- Note: Explain results are often viewed directly but not typically stored in a persistent table for the app.
-- We will still run the query here for completeness of the SQL script.
-- =================================================================================================

SELECT
  *
FROM
  ML.EXPLAIN_FORECAST(MODEL `${project_id}.${dataset_id}.arima_ga_daily_revenue_model`,
                      STRUCT(120 AS horizon, 0.95 AS confidence_level));

-- =================================================================================================
-- SECTION 6: FORECAST WITH TimesFM 2.0 MODEL & COMPARE
-- =================================================================================================
-- Purpose: To generate forecasts using TimesFM 2.0 and compare them with ARIMA_PLUS.
-- =================================================================================================

-- Step 6.1: Perform and view the TimesFM forecast directly (optional view).
SELECT
  *
FROM
  AI.FORECAST(
    TABLE `${project_id}.${dataset_id}.ga_daily_revenue`,
    data_col => 'total_daily_revenue',
    timestamp_col => 'transaction_date',
    model => 'TimesFM 2.0',
    horizon => 120,
    confidence_level => 0.95
  )
ORDER BY
  forecast_timestamp;

-- Step 6.1.1: Create a persistent table for TimesFM Forecast results
CREATE OR REPLACE TABLE `${project_id}.${dataset_id}.ga_timesfm_forecast` AS
SELECT
  *
FROM
  AI.FORECAST(
    TABLE `${project_id}.${dataset_id}.ga_daily_revenue`,
    data_col => 'total_daily_revenue',
    timestamp_col => 'transaction_date',
    model => 'TimesFM 2.0',
    horizon => 120,
    confidence_level => 0.95
  );

-- Step 6.2: Create a comprehensive table combining actuals, forecasts, and anomalies.
-- This single table simplifies data fetching for the application.
CREATE OR REPLACE TABLE `${project_id}.${dataset_id}.ga_combined_forecast_data` AS
WITH
  Actuals AS (
    SELECT
      transaction_date,
      total_daily_revenue AS actual_revenue
    FROM
      `${project_id}.${dataset_id}.ga_daily_revenue`
  ),
  ArimaForecast AS (
    SELECT
      CAST(forecast_timestamp AS DATE) AS forecast_date,
      forecast_value AS arima_forecast_value,
      prediction_interval_lower_bound AS arima_lower_bound,
      prediction_interval_upper_bound AS arima_upper_bound
    FROM
      `${project_id}.${dataset_id}.ga_arima_forecast`
  ),
  TimesfmForecast AS (
    -- Read from the pre-calculated TimesFM forecast table
    SELECT
      CAST(forecast_timestamp AS DATE) AS forecast_date,
      forecast_value AS timesfm_forecast_value,
      prediction_interval_lower_bound AS timesfm_lower_bound,
      prediction_interval_upper_bound AS timesfm_upper_bound
    FROM
      `${project_id}.${dataset_id}.ga_timesfm_forecast` -- Read from the new table
  ),
  Anomalies AS (
    -- Casting the likely TIMESTAMP output to DATE for joining
    SELECT
      CAST(transaction_date AS DATE) AS anomaly_date, -- CASTING to DATE
      total_daily_revenue AS anomaly_value,
      is_anomaly
    FROM
      `${project_id}.${dataset_id}.ga_arima_anomalies`
    WHERE is_anomaly = TRUE
  )
-- Combine using FULL OUTER JOIN on date
SELECT
  COALESCE(a.transaction_date, af.forecast_date, tf.forecast_date, anom.anomaly_date) AS report_date,
  a.actual_revenue,
  af.arima_forecast_value,
  af.arima_lower_bound,
  af.arima_upper_bound,
  tf.timesfm_forecast_value,
  tf.timesfm_lower_bound,
  tf.timesfm_upper_bound,
  anom.is_anomaly IS NOT NULL AS is_anomaly,
  anom.anomaly_value -- Value at the time of the anomaly
FROM Actuals a
FULL OUTER JOIN ArimaForecast af ON a.transaction_date = af.forecast_date
FULL OUTER JOIN TimesfmForecast tf ON COALESCE(a.transaction_date, af.forecast_date) = tf.forecast_date
FULL OUTER JOIN Anomalies anom ON COALESCE(a.transaction_date, af.forecast_date, tf.forecast_date) = anom.anomaly_date -- Now comparing DATE = DATE
ORDER BY
  report_date;

-- =================================================================================================
-- SECTION 7: CONTRIBUTION ANALYSIS MODEL (GA Revenue Drivers) & STORE INSIGHTS
-- =================================================================================================
-- Purpose: To identify drivers of change in daily revenue and store the insights.
-- =================================================================================================

-- Step 7.1: Create the Contribution Analysis model
CREATE OR REPLACE MODEL `${project_id}.${dataset_id}.ga_revenue_contribution_model`
OPTIONS(
  MODEL_TYPE = 'CONTRIBUTION_ANALYSIS',
  CONTRIBUTION_METRIC = 'SUM(daily_revenue)',
  DIMENSION_ID_COLS = ['deviceCategory', 'country', 'channelGrouping'],
  IS_TEST_COL = 'is_recent_period'
) AS
WITH PreprocessedGAData AS (
  SELECT
    (totals.transactionRevenue / 1000000) AS daily_revenue,
    device.deviceCategory,
    geoNetwork.country,
    channelGrouping,
    PARSE_DATE('%Y%m%d', date) >= DATE('2017-07-01') AS is_recent_period
  FROM
    `bigquery-public-data.google_analytics_sample.ga_sessions_*`
  WHERE
    _TABLE_SUFFIX BETWEEN '20170601' AND '20170731'
    AND totals.transactionRevenue IS NOT NULL
    AND device.deviceCategory IS NOT NULL
    AND geoNetwork.country IS NOT NULL
    AND channelGrouping IS NOT NULL
)
SELECT
  daily_revenue,
  deviceCategory,
  country,
  channelGrouping,
  is_recent_period
FROM
  PreprocessedGAData;

-- Step 7.2: Query the Contribution Analysis model and store insights
CREATE OR REPLACE TABLE `${project_id}.${dataset_id}.ga_contribution_insights` AS
SELECT *
FROM ML.GET_INSIGHTS(MODEL `${project_id}.${dataset_id}.ga_revenue_contribution_model`);


-- =================================================================================================
-- SECTION 8: REGISTER ARIMA MODEL TO VERTEX AI MODEL REGISTRY
-- =================================================================================================
-- Purpose: To export the trained BQML ARIMA model for GA Revenue to Vertex AI.
-- =================================================================================================

ALTER MODEL IF EXISTS `${project_id}.${dataset_id}.arima_ga_daily_revenue_model`
SET OPTIONS (vertex_ai_model_id="arima_ga_daily_revenue_model");

-- =================================================================================================
-- END OF SCRIPT
-- ================================================================================================= 