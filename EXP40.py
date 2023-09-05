import pandas as pd
import matplotlib.pyplot as plt

csv_file = 'player_data.csv'  
data = pd.read_csv(csv_file)

top_goal_scorers = data.nlargest(5, 'Goals')

top_salary_players = data.nlargest(5, 'Salary')

average_age = data['Age'].mean()

above_average_age_players = data[data['Age'] > average_age]['Name']

position_counts = data['Position'].value_counts()
plt.bar(position_counts.index, position_counts.values)
plt.xlabel('Position')
plt.ylabel('Number of Players')
plt.title('Distribution of Players Based on Positions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Top 5 players with the highest number of goals:")
print(top_goal_scorers[['Name', 'Goals']])

print("\nTop 5 players with the highest salaries:")
print(top_salary_players[['Name', 'Salary']])

print(f"\nAverage age of players: {average_age:.2f}")

print("\nPlayers above the average age:")
print(above_average_age_players)
