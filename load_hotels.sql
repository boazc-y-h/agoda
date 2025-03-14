TRUNCATE TABLE dbo.hotels
 
-- import the file
BULK INSERT dbo.hotels
FROM 'C:\Users\user\source\repos\agoda\data\hotels.csv'
WITH
(
        FORMAT='CSV',
        FIRSTROW=2
)