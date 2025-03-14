-- number of booking and average price by flight type
SELECT flightType, COUNT(*) AS number_of_bookings, ROUND(AVG(price),2) AS avg_price
FROM flights
GROUP BY flightType;