from app import create_app
from flask import jsonify, request
import sqlalchemy
from sqlalchemy.sql import text
import os
import data_analysis as da

#ENV variable
localhost=True
fields = ['team_name','points','yards','pass_yards','rush_yards']

conn = sqlalchemy.create_engine('sqlite:///football_stats.db')
app = create_app("development")

if(not localhost):
    import logging
    logging.basicConfig(filename='logging.log',level=logging.DEBUG)

def map_keys_to_values(keys, values):
    return {key : value for key, value in zip(keys, values)}

@app.route('/')
def index():
    return 'To query the API, put team names as parameters. For example: "/team_a=DEN&team_b=DET" '

@app.route('/to_sql')
def sqlify():
    if(localhost):
        da.to_sql()
        return "Done!"
    else:
        return "This can only be done locally!"

@app.route('/team_a=<a>&team_b=<b>')
def get_stats(a,b):
    return jsonify(da.stats(a,b))

@app.route('/max_difference/<stat>')
def get_max_difference(stat):
    if stat not in fields:
        return {"error":"Couldn't query database - invalid stat."}

    max_offense = conn.execute(text("select max("+stat+") from offense")).fetchone()[0]
    min_offense = conn.execute(text("select min("+stat+") from offense")).fetchone()[0]

    max_defense = conn.execute(text("select max("+stat+") from defense")).fetchone()[0]
    min_defense = conn.execute(text("select min("+stat+") from defense")).fetchone()[0]

    output = {}
    output["offense"] = max_offense-min_offense
    output["defense"] = max_defense-min_defense
    return jsonify(output)

@app.route('/stats/<team>')
def get_team_stats(team):
    output = {}
    query = text("select "+','.join(fields)+" from defense where team_name = :t")
    defense_stats = conn.execute(query,t=team).fetchone()
    output['defense'] = map_keys_to_values(fields,defense_stats)
    
    query = text("select "+','.join(fields)+" from offense where team_name = :t")
    offense_stats = conn.execute(query,t=team).fetchone()
    output['offense'] = map_keys_to_values(fields,offense_stats)
    
    return jsonify(output)

@app.route('/log')
def read_log():
    f  = open('logging.log', "r")
    return f.read()


@app.route('/status')
def status():
    return 'Ok!'

@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response

if __name__ == '__main__':
    if(localhost):
        app.run()
    else:
        port = int(os.environ.get("PORT", 8886))
        app.run(host='0.0.0.0', port=port)