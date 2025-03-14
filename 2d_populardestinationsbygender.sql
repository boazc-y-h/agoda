-- booking by destination and gender
SELECT flights.[to], users.gender, COUNT(*) AS bookings
FROM flights
INNER JOIN users
ON flights.userCode = users.code
GROUP BY flights.[to], users.gender
ORDER BY 2,3 DESC;