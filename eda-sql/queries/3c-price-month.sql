-- avg price by months
SELECT a.travel_month, a.avg_flight_price, b.avg_hotel_price
FROM (
SELECT EXTRACT(MONTH FROM date) AS travel_month, ROUND(AVG(price)::NUMERIC,2) AS avg_flight_price
FROM flights
GROUP BY EXTRACT(MONTH FROM date)
    ) AS a
LEFT JOIN (
	SELECT EXTRACT(MONTH FROM date) AS travel_month, ROUND(AVG(price)::NUMERIC,2) as avg_hotel_price
	FROM hotels
	GROUP BY EXTRACT(MONTH FROM date)
	) AS b
ON a.travel_month = b.travel_month
ORDER BY 2 DESC;
