 
-- import the file
BULK INSERT dbo.flights
FROM 'C:\tmp\flights.csv'
WITH
(
        FORMAT='CSV',
        FIRSTROW=2
)