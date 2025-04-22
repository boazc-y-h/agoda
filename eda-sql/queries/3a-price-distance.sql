-- price by distance
SELECT 
	CONCAT(FLOOR(distance/100)*100,'-',FLOOR(distance/100)*100+99) AS flight_distance,
	ROUND(AVG(price)::NUMERIC,2) AS average_price
FROM flights
GROUP BY FLOOR(distance/100)*100
ORDER BY 1;