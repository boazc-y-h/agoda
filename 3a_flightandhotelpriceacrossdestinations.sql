-- average flight and hotel price across destinations
SELECT a.destination, a.avg_flight_price, b.avg_hotel_price
FROM (
	SELECT [to] as destination, ROUND(AVG(price),2) as avg_flight_price
	FROM flights
	GROUP BY [to]
	) AS a
LEFT JOIN (
	SELECT place, ROUND(AVG(price),2) as avg_hotel_price
	FROM hotels
	GROUP BY place
	) AS b
ON a.destination = b.place