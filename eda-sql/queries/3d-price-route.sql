-- flight bookings and avg price by flight route
SELECT "from", "to", COUNT(*) AS bookings, ROUND(AVG(price)::NUMERIC,2) AS avg_price
FROM flights
GROUP BY "from", "to"
ORDER BY 4 DESC
LIMIT 10;