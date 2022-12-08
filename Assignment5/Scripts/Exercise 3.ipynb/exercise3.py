import pandas as pd
import json

data = pd.read_csv("cleaned_cancer_data.csv")

age_adjusted_IR = data['Age-Adjusted Incidence Rate([rate note]) - cases per 100,000'].tolist()
state_name1 = data['State'].tolist()
state_name = []
for state in state_name1:
    state_name.append(state.lower())

from flask import Flask, render_template, request, url_for
app = Flask(__name__)

@app.route("/")
def homepage():
    return render_template("index.html")

@app.route("/state/<string:name>")
def get_info():
    dics = {}
    for item in state_name:
        dics[item] = age_adjusted_IR[state_name.index(item)]
    json_object = json.dumps(dics)
    return json_object

@app.route("/info", methods=["GET"])
def info():
    UserState = request.args.get("UserState")
    result = ""
    IR = 0
    FIPS = 0
    CI = ""
    AAC = 0
    rec_trend = ""
    if UserState.lower().strip() in str(data['State']).lower():
        i=0
        for item in data['State']: 
            print(item)
            if UserState.lower().strip() == item.lower():
                FIPS = data.iloc[i,2]
                print(FIPS)
                IR = data.iloc[i,3]
                print(IR)
                CI = "(" + str(data.iloc[i,4]) + "," + str(data.iloc[i,5]) + ")"
                AAC = int(data.iloc[i,6])
                rec_trend = data.iloc[i,7]
            i+=1
        result += f"The state name is {UserState}, the age-adjusted incidence rate(cases per 100k) is {IR}.\n" 
        print(result)
        return render_template("info.html", analysis = result, FIPS = FIPS, IR = IR, CI = CI, AAC = AAC, rec_trend = rec_trend, usertext = UserState)
    else:
        result += f"Error: the state name {UserState} is invalid.\n"
        return render_template("error.html", analysis = result, usertext = UserState)

if __name__ == "__main__":
    app.run(debug = True)