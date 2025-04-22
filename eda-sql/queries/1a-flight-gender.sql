-- flight bookings contributed by different genders
SELECT 
    gender, COUNT(*) AS flight_count
FROM 
    flights
INNER JOIN 
    users 
ON 
    flights."userCode" = users."code" 
GROUP BY 
    gender;