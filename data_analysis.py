import pandas as pd
import numpy as np
from run import conn

df_games = pd.read_sql_query("select * from games;", conn)
df_offense = pd.read_sql_query("select * from offense;", conn)
df_defense = pd.read_sql_query("select * from defense;", conn)

def to_sql():
    df_games.to_sql("games", conn, if_exists="replace")
    df_offense.to_sql("offense", conn, if_exists="replace")
    df_defense.to_sql("defense", conn, if_exists="replace")

#Find the distance between two points
#The distance is a good indication on how different two teams are
#Find which team is preferred based on the distances from the four stats
def euclidean_coords(x_val, y_val,team_a, team_b):
    #TODO: Have another dataframe mapping acronym names with full names
    i_a = df_offense.index[df_offense['team_name'] == team_a][0]
    i_b = df_offense.index[df_offense['team_name'] == team_b][0]
    a = np.array((float(x_val[i_a]),float(y_val[i_a])))
    b = np.array((float(x_val[i_b]),float(y_val[i_b])))
    return a-b
#Used to find euclidean distance
#dist = np.linalg.norm(a-b)

# 'X' has to be minimized and 'Y' has to be maximized for A to be better than B.
def stats(a, b):
    stats_dict = {}
    stats_dict['points_coords'] = euclidean_coords(df_defense["points"], df_offense["points"],a,b).tolist() #Distance for points
    stats_dict['yards_coords'] = euclidean_coords(df_defense["yards"], df_offense["yards"],a,b).tolist() #Distance for yards
    stats_dict['pass_coords'] = euclidean_coords(df_defense["pass_yards"], df_offense["pass_yards"],a,b).tolist() #Distance for yards
    stats_dict['rush_coords'] = euclidean_coords(df_defense["rush_yards"], df_offense["rush_yards"],a,b).tolist() #Distance for yards

    return stats_dict
