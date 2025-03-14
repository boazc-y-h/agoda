SELECT 
	FLOOR(distance/100)*100 AS flight_distance,
	ROUND(AVG(price),2) AS avg_price
FROM flights
GROUP BY FLOOR(distance/100)*100
ORDER BY 1;