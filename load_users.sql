 
-- import the file
BULK INSERT dbo.users
FROM 'C:\tmp\users.csv'
WITH
(
        FORMAT='CSV',
        FIRSTROW=2
)