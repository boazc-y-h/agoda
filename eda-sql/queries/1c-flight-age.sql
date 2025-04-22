-- flights by age group
SELECT 
    CONCAT(floor(age/10)*10,'-',floor(age/10)*10+9) AS age_group, 
    COUNT(*) AS number_of_bookings, 
    ROUND(AVG(price)::NUMERIC,2) AS average_price
FROM flights 
INNER JOIN users 
ON flights."userCode" = users.code
GROUP BY floor(age/10)*10
ORDER BY 1;