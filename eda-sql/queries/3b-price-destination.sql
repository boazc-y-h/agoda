-- average prices by destinations
SELECT a.destination, a.avg_flight_price, b.avg_hotel_price
FROM (
	SELECT flights.to as destination, ROUND(AVG(price)::NUMERIC,2) as avg_flight_price
	FROM flights
	GROUP BY flights.to
	) AS a
LEFT JOIN (
	SELECT place, ROUND(AVG(price)::NUMERIC,2) as avg_hotel_price
	FROM hotels
	GROUP BY place
	) AS b
ON a.destination = b.place
ORDER BY avg_flight_price DESC, avg_hotel_price DESC;