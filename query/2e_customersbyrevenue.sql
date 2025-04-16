SELECT users.name, c.flight_revenue, c.hotel_revenue, c.total_revenue
FROM (
	SELECT a.userCode, 
		a.flight_revenue, 
		ISNULL(b.hotel_revenue, 0) AS hotel_revenue, 
		a.flight_revenue+ISNULL(b.hotel_revenue, 0) AS total_revenue
	FROM (
		SELECT userCode, SUM(price) AS flight_revenue
		FROM flights
		GROUP BY userCode
		) AS a
	FULL OUTER JOIN (
		SELECT userCode, SUM(total) AS hotel_revenue
		FROM hotels
		GROUP BY userCode
		) AS b
	ON a.userCode = b.userCode
	) AS c
INNER JOIN users 
ON c.userCode = users.code
ORDER BY 4 DESC;