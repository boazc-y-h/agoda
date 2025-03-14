-- total bookings by months
SELECT a.travel_month, a.flight_bookings, b.hotel_bookings, a.flight_bookings+b.hotel_bookings AS total_bookings
FROM (
	SELECT MONTH(date) AS travel_month, COUNT(*) AS flight_bookings
	FROM flights
	GROUP BY MONTH(date)
	) AS a
INNER JOIN (
	SELECT MONTH(date) AS travel_month, COUNT(*) AS hotel_bookings
	FROM hotels
	GROUP BY MONTH(date)
	) AS b
ON a.travel_month = b.travel_month
ORDER BY 4 DESC;