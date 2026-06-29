import pandas as pd
from sqlalchemy import create_engine

# Connect to MySQL
engine = create_engine('mysql+pymysql://root:YOUR_PASSWORD@localhost/car_listing')

# Load CSV
df = pd.read_csv('/PATH_TO/vehicles.csv')

# Import to MySQL
df.to_sql('listings', con=engine, if_exists='replace', index=False)

print("Done!")
