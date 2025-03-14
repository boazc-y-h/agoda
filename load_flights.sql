TRUNCATE TABLE dbo.flights
 
-- import the file
BULK INSERT dbo.flights
FROM 'C:\Users\user\source\repos\agoda\data\flights.csv'
WITH
(
        FORMAT='CSV',
        FIRSTROW=2
)