-- popular destinations by age group
SELECT 
    flights.to, 
    CONCAT(floor(age/10)*10,'-',floor(age/10)*10+9) AS age_group, 
    COUNT(*) AS bookings
FROM flights
INNER JOIN users
ON flights."userCode" = users.code
GROUP BY flights.to, floor(age/10)*10
ORDER BY 2,3 DESC;