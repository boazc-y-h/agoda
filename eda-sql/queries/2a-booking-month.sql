-- bookings by months
SELECT a.travel_month, a.flight_bookings, b.hotel_bookings, a.flight_bookings+b.hotel_bookings AS total_bookings
FROM (
	SELECT EXTRACT(MONTH FROM date) AS travel_month, COUNT(*) AS flight_bookings
	FROM flights
	GROUP BY EXTRACT(MONTH FROM date)
	) AS a
INNER JOIN (
	SELECT EXTRACT(MONTH FROM date) AS travel_month, COUNT(*) AS hotel_bookings
	FROM hotels
	GROUP BY EXTRACT(MONTH FROM date)
	) AS b
ON a.travel_month = b.travel_month
ORDER BY 4 DESC;