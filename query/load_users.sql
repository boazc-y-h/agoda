TRUNCATE TABLE dbo.users

-- import the file
BULK INSERT dbo.users
FROM 'C:\Users\user\source\repos\agoda\data\users.csv'
WITH
(
        FORMAT='CSV',
        FIRSTROW=2
)