-- average prices by agency
SELECT f.agency, 
       ROUND(AVG(f.price)::NUMERIC, 2) AS avg_flight_price, 
       ROUND(AVG(h.price)::NUMERIC, 2) AS avg_hotel_price
FROM flights f
LEFT JOIN hotels h ON f."travelCode" = h."travelCode"
GROUP BY f.agency;