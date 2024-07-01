import pandas as pd

# Path to your CSV file
path = r"C:\Users\fabbr\Desktop\Dev\Data Science\Lol-Oracle\team_data.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(path)

# Filter the DataFrame for the specific game ID
df = df[df["gameid"] == "LOLTMNT05_57443"]

# Extract the towers and total_towers_in_game columns
towers = df["towers"]
total_towers = df["total_towers_in_game"]

# Print the values using the values attribute
print(towers.values)
print(total_towers.values)

# Path to your CSV file
path2 = r"C:\Users\fabbr\Desktop\Dev\Data Science\Lol-Oracle\player_data.csv"

# Read the CSV file into a DataFrame
df2 = pd.read_csv(path2)

# Filter the DataFrame for the specific game ID
df2 = df2[df2["gameid"] == "LOLTMNT05_57443"]

# Extract the towers and total_towers_in_game columns
ka_ratio = df2["ka_ratio"]

# Print the values using the values attribute
print(ka_ratio)
