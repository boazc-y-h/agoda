-- flight bookings contributed by age
SELECT floor(age/10)*10 AS bin_floor, COUNT(*) AS number_of_bookings, COUNT(*)*100/(SELECT COUNT(*) FROM flights)
FROM flights 
INNER JOIN users 
ON flights.userCode = users.code
GROUP BY floor(age/10)*10
ORDER BY 1;