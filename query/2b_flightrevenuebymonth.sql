SELECT a.travel_month, a.flight_revenue, b.hotel_revenue, a.flight_revenue+b.hotel_revenue AS total_revenue
FROM (
	SELECT MONTH(date) AS travel_month, ROUND(SUM(price),2) AS flight_revenue
	FROM flights
	GROUP BY MONTH(date)
	) AS a
INNER JOIN (
	SELECT MONTH(date) AS travel_month, ROUND(SUM(total),2) AS hotel_revenue
	FROM hotels
	GROUP BY MONTH(date)
	) AS b
ON a.travel_month = b.travel_month
ORDER BY 1;