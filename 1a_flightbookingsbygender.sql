-- flight bookings contributed by different genders
SELECT gender, COUNT(*) AS number_of_bookings, COUNT(*)*100/(SELECT COUNT(*) FROM flights)
FROM flights 
INNER JOIN users 
ON flights.userCode = users.code
GROUP BY gender;