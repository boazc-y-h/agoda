-- total bookings and avg price by months

SELECT MONTH(date) AS travel_month, COUNT(*) AS flight_bookings, ROUND(AVG(price),2) AS avg_flight_price
FROM flights
GROUP BY MONTH(date)
ORDER BY 1
