 
-- import the file
BULK INSERT dbo.hotels
FROM 'C:\tmp\hotels.csv'
WITH
(
        FORMAT='CSV',
        FIRSTROW=2
)