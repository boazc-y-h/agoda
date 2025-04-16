-- avg price of flight bookings by age
SELECT floor(age/10)*10 AS bin_floor, ROUND(AVG(price),2) AS avg_price
FROM flights 
INNER JOIN users 
ON flights.userCode = users.code
GROUP BY floor(age/10)*10
ORDER BY 1;