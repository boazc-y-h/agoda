-- revenue by month
SELECT a.travel_month, a.flight_revenue, b.hotel_revenue, ROUND(a.flight_revenue + b.hotel_revenue,2) AS total_revenue
FROM (
    SELECT EXTRACT(MONTH FROM date) AS travel_month, ROUND(SUM(price)::NUMERIC, 2) AS flight_revenue
    FROM flights
    GROUP BY EXTRACT(MONTH FROM date)
) AS a
INNER JOIN (
    SELECT EXTRACT(MONTH FROM date) AS travel_month, ROUND(SUM(total)::NUMERIC, 2) AS hotel_revenue
    FROM hotels
    GROUP BY EXTRACT(MONTH FROM date)
) AS b
ON a.travel_month = b.travel_month
ORDER BY total_revenue DESC;