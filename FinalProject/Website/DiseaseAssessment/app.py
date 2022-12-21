import plotly.graph_objs as go
from flask import Flask
from flask import Flask, render_template, request
from flask import render_template_string, jsonify
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from json import dumps
from plotly import utils
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import json
import plotly
import plotly.express as px

app = Flask(__name__)

dataset = pd.read_csv('./Dataset2BRFSS2015.csv')
dataset_copy = dataset.copy(deep=True)
dataset_copy.Diabetes_binary[dataset_copy['Diabetes_binary']
                             == 0] = 'No Diabetes'
dataset_copy.Diabetes_binary[dataset_copy['Diabetes_binary'] == 1] = 'Diabetes'

dataset_copy.HighBP[dataset_copy['HighBP'] == 0] = 'No High BP'
dataset_copy.HighBP[dataset_copy['HighBP'] == 1] = 'High BP'

dataset_copy.HighChol[dataset_copy['HighChol'] == 0] = 'No High Cholesterol'
dataset_copy.HighChol[dataset_copy['HighChol'] == 1] = 'High Cholesterol'

dataset_copy.CholCheck[dataset_copy['CholCheck']
                       == 0] = 'No Cholesterol Check in 5 Years'
dataset_copy.CholCheck[dataset_copy['CholCheck']
                       == 1] = 'Cholesterol Check in 5 Years'

dataset_copy.Smoker[dataset_copy['Smoker'] == 0] = 'No'
dataset_copy.Smoker[dataset_copy['Smoker'] == 1] = 'Yes'

dataset_copy.Stroke[dataset_copy['Stroke'] == 0] = 'No'
dataset_copy.Stroke[dataset_copy['Stroke'] == 1] = 'Yes'

dataset_copy.HeartDiseaseorAttack[dataset_copy['HeartDiseaseorAttack'] == 0] = 'No'
dataset_copy.HeartDiseaseorAttack[dataset_copy['HeartDiseaseorAttack'] == 1] = 'Yes'

dataset_copy.PhysActivity[dataset_copy['PhysActivity'] == 0] = 'No'
dataset_copy.PhysActivity[dataset_copy['PhysActivity'] == 1] = 'Yes'

dataset_copy.Fruits[dataset_copy['Fruits'] == 0] = 'No'
dataset_copy.Fruits[dataset_copy['Fruits'] == 1] = 'Yes'

dataset_copy.Veggies[dataset_copy['Veggies'] == 0] = 'No'
dataset_copy.Veggies[dataset_copy['Veggies'] == 1] = 'Yes'

dataset_copy.HvyAlcoholConsump[dataset_copy['HvyAlcoholConsump'] == 0] = 'No'
dataset_copy.HvyAlcoholConsump[dataset_copy['HvyAlcoholConsump'] == 1] = 'Yes'

dataset_copy.AnyHealthcare[dataset_copy['AnyHealthcare'] == 0] = 'No'
dataset_copy.AnyHealthcare[dataset_copy['AnyHealthcare'] == 1] = 'Yes'

dataset_copy.NoDocbcCost[dataset_copy['NoDocbcCost'] == 0] = 'No'
dataset_copy.NoDocbcCost[dataset_copy['NoDocbcCost'] == 1] = 'Yes'

dataset_copy.GenHlth[dataset_copy['GenHlth'] == 1] = 'Excellent'
dataset_copy.GenHlth[dataset_copy['GenHlth'] == 2] = 'Very Good'
dataset_copy.GenHlth[dataset_copy['GenHlth'] == 3] = 'Good'
dataset_copy.GenHlth[dataset_copy['GenHlth'] == 4] = 'Fair'
dataset_copy.GenHlth[dataset_copy['GenHlth'] == 5] = 'Poor'

dataset_copy.DiffWalk[dataset_copy['DiffWalk'] == 0] = 'No'
dataset_copy.DiffWalk[dataset_copy['DiffWalk'] == 1] = 'Yes'

dataset_copy.Sex[dataset_copy['Sex'] == 0] = 'Female'
dataset_copy.Sex[dataset_copy['Sex'] == 1] = 'Male'

dataset_copy.Age[dataset_copy['Age'] == 1] = '18 to 24'
dataset_copy.Age[dataset_copy['Age'] == 2] = '25 to 29'
dataset_copy.Age[dataset_copy['Age'] == 3] = '30 to 34'
dataset_copy.Age[dataset_copy['Age'] == 4] = '35 to 39'
dataset_copy.Age[dataset_copy['Age'] == 5] = '40 to 44'
dataset_copy.Age[dataset_copy['Age'] == 6] = '45 to 49'
dataset_copy.Age[dataset_copy['Age'] == 7] = '50 to 54'
dataset_copy.Age[dataset_copy['Age'] == 8] = '55 to 59'
dataset_copy.Age[dataset_copy['Age'] == 9] = '60 to 64'
dataset_copy.Age[dataset_copy['Age'] == 10] = '65 to 69'
dataset_copy.Age[dataset_copy['Age'] == 11] = '70 to 74'
dataset_copy.Age[dataset_copy['Age'] == 12] = '75 to 79'
dataset_copy.Age[dataset_copy['Age'] == 13] = '80 or older'

dataset_copy.Education[dataset_copy['Education']
                       == 1] = 'Never Attended School'
dataset_copy.Education[dataset_copy['Education'] == 2] = 'Elementary'
dataset_copy.Education[dataset_copy['Education'] == 3] = 'Junior High School'
dataset_copy.Education[dataset_copy['Education'] == 4] = 'Senior High School'
dataset_copy.Education[dataset_copy['Education'] == 5] = 'Undergraduate Degree'
dataset_copy.Education[dataset_copy['Education'] == 6] = 'Magister'

dataset_copy.Income[dataset_copy['Income'] == 1] = 'Less Than $10,000'
dataset_copy.Income[dataset_copy['Income']
                    == 2] = 'Between $10,000 and $15,000'
dataset_copy.Income[dataset_copy['Income']
                    == 3] = 'Between $15,000 and $20,000'
dataset_copy.Income[dataset_copy['Income']
                    == 4] = 'Between $20,000 and $25,000'
dataset_copy.Income[dataset_copy['Income']
                    == 5] = 'Between $25,000 and $35,000'
dataset_copy.Income[dataset_copy['Income']
                    == 6] = 'Between $35,000 and $50,000'
dataset_copy.Income[dataset_copy['Income']
                    == 7] = 'Between $50,000 and $75,000'
dataset_copy.Income[dataset_copy['Income'] == 8] = '$75,000 or More'
summary_cat = dataset_copy.describe(include='object', exclude='float64')
summary_cat_transposed = summary_cat.T
summary_num = dataset_copy.describe(include='float64', exclude='object')
summary_num_transposed = summary_num.T

DiabeticSet = dataset_copy.loc[dataset_copy['Diabetes_binary'] == "Diabetes"]
summary_cat_diabetes = DiabeticSet.describe(
    include='object', exclude='float64')
summary_cat_diabetes_transposed = summary_cat_diabetes.T

summary_num_diabetes = DiabeticSet.describe(
    include='float64', exclude='object')
summary_num_diabetes_transposed = summary_num_diabetes.T

DiabeticSet = dataset_copy.loc[dataset_copy['Diabetes_binary']
                               == "No Diabetes"]
summary_cat_Nodiabetes = DiabeticSet.describe(
    include='object', exclude='float64')
summary_cat_Nodiabetes_transposed = summary_cat_Nodiabetes.T

summary_num_Nodiabetes = DiabeticSet.describe(
    include='float64', exclude='object')
summary_num_Nodiabetes_transposed = summary_num_Nodiabetes.T

bivariate_plot=px.histogram(dataset_copy, x=dataset_copy['BMI'], title='BMI vs. Diabetes', color='Diabetes_binary')
        
data_selected = dataset.loc[:, ['Diabetes_binary', 'HighBP', 'BMI', 'PhysHlth', 'GenHlth', 'MentHlth',
                                'Age', 'Education', 'Income', 'Smoker', 'Sex']]

def graphPlot(variable):
    if variable == "BMI":
        bivariate_plot=px.histogram(dataset_copy, x=dataset_copy['BMI'], title='BMI vs. Diabetes', color='Diabetes_binary')
    if variable == "HighBP":
        diabetes_bp = dataset_copy.groupby(["Diabetes_binary", "HighBP"]).size().reset_index(name = "Count")
        bivariate_plot=px.bar(diabetes_bp, x="HighBP", y="Count", color="Diabetes_binary", title="Diabates-HighBP Distribution")
    if variable == "HighChol":
        diabetes_chol = dataset_copy.groupby(['Diabetes_binary', 'HighChol']).size().reset_index(name = 'Count')
        bivariate_plot = px.bar(diabetes_chol, x="HighChol", y="Count", color="Diabetes_binary", title="Diabates-HighChol Distribution")
    if variable == "CholCheck":
        CholCheck = dataset_copy.groupby(['Diabetes_binary', 'CholCheck']).size().reset_index(name = 'Count')
        bivariate_plot = px.bar(CholCheck, x="CholCheck", y="Count", color="Diabetes_binary", title="Diabates-CholCheck Distribution")
    if variable == "Smoker":
        Smoker = dataset_copy.groupby(['Diabetes_binary', 'Smoker']).size().reset_index(name = 'Count')
        bivariate_plot = px.bar(Smoker, x="Smoker", y="Count", color="Diabetes_binary", title="Diabates-Smoking Distribution")
    if variable == "Stroke":
        Stroke = dataset_copy.groupby(['Diabetes_binary', 'Stroke']).size().reset_index(name = 'Count')
        bivariate_plot = px.bar(Stroke, x="Stroke", y="Count", color="Diabetes_binary", title="Diabates-StrokeDistribution")
    if variable == "Age":
        Age = dataset_copy.groupby(['Diabetes_binary', 'Age']).size().reset_index(name = 'Count')
        bivariate_plot = px.bar(Age, x="Age", y="Count", color="Diabetes_binary", title="Diabates-Age Distribution")
    if variable == "Education":  
        Education = dataset_copy.groupby(['Diabetes_binary', 'Education']).size().reset_index(name = 'Count')
        bivariate_plot = px.bar(Education , x="Education", y="Count", color="Diabetes_binary", title="Diabates-Education Distribution")
    if variable == "Income": 
        Income = dataset_copy.groupby(['Diabetes_binary', 'Income']).size().reset_index(name = 'Count')
        bivariate_plot = px.bar(Income , x="Income", y="Count", color="Diabetes_binary", title="Diabates-Income Distribution")
    if variable == "Fruits": 
        Fruits = dataset_copy.groupby(['Diabetes_binary', 'Fruits']).size().reset_index(name = 'Count')
        bivariate_plot = px.bar(Fruits, x="Fruits", y="Count", color="Diabetes_binary", title="Diabates-Fruits Intake Distribution")
    if variable == "Veggie": 
        Veggies = dataset_copy.groupby(['Diabetes_binary', 'Veggies']).size().reset_index(name = 'Count')
        bivariate_plot = px.bar(Veggies, x="Veggies", y="Count", color="Diabetes_binary", title="Diabates-Veggies Intake Distribution")
    if variable == "PhysActivity": 
        PhysActivity = dataset_copy.groupby(['Diabetes_binary', 'PhysActivity']).size().reset_index(name = 'Count')
        bivariate_plot = px.bar(PhysActivity, x="PhysActivity", y="Count", color="Diabetes_binary", title="Diabates-PhysicalActivity Distribution")
    if variable == "HvyAlcoholConsump": 
        HvyAlcoholConsump = dataset_copy.groupby(['Diabetes_binary', 'HvyAlcoholConsump']).size().reset_index(name = 'Count')
        bivariate_plot = px.bar(HvyAlcoholConsump , x="HvyAlcoholConsump", y="Count", color="Diabetes_binary", title="Diabates-HvyAlcoholConsump Distribution")
    if variable == "AnyHealthcare": 
        AnyHealthcare = dataset_copy.groupby(['Diabetes_binary', 'AnyHealthcare']).size().reset_index(name = 'Count')
        bivariate_plot = px.bar(AnyHealthcare , x="AnyHealthcare", y="Count", color="Diabetes_binary", title="Diabates-AnyHealthcare Distribution")
    if variable == "GenHlth": 
        GenHlth = dataset_copy.groupby(['Diabetes_binary', 'GenHlth']).size().reset_index(name = 'Count')
        bivariate_plot = px.bar(GenHlth, x="GenHlth", y="Count", color="Diabetes_binary", title="Diabates-GeneralHealth Distribution")
    if variable == "DiffWalk": 
        DiffWalk = dataset_copy.groupby(['Diabetes_binary', 'DiffWalk']).size().reset_index(name = 'Count')
        bivariate_plot = px.bar(DiffWalk, x="DiffWalk", y="Count", color="Diabetes_binary", title="Diabates-DiffWalk Distribution")
    if variable == "Sex": 
        Gender = dataset_copy.groupby(['Diabetes_binary', 'Sex']).size().reset_index(name = 'Count')
        bivariate_plot = px.bar(Gender, x="Sex", y="Count", color="Diabetes_binary", title="Diabates-Gender Distribution")
    if variable == "NoDocbcCost": 
        NoDocbcCost = dataset_copy.groupby(['Diabetes_binary', 'NoDocbcCost']).size().reset_index(name = 'Count')
        bivariate_plot = px.bar(NoDocbcCost, x="NoDocbcCost", y="Count", color="Diabetes_binary", title="Diabates-NoDocbcCost Distribution")
    if variable == "HeartDiseaseorAttack": 
        HeartDiseaseorAttack = dataset_copy.groupby(['Diabetes_binary', 'HeartDiseaseorAttack']).size().reset_index(name = 'Count')
        bivariate_plot = px.bar(HeartDiseaseorAttack, x="HeartDiseaseorAttack", y="Count", color="Diabetes_binary", title="Diabates-HeartDisease/Attack Distribution")
    if variable == "MentHlth": 
        bivariate_plot = px.box(dataset_copy, x="MentHlth", color="Diabetes_binary")
    if variable == "PhysHlth": 
        bivariate_plot = px.box(dataset_copy, x="PhysHlth", color="Diabetes_binary")
    return bivariate_plot

def univariateGraphPlot(variable):
    if variable == "Diabetes":
        Diabetes_status = dataset_copy.Diabetes_binary.value_counts().reset_index(name = 'Count')
        univariate_plot =px.bar(Diabetes_status, x='index', y = "Count", title = "Diabetes Status", 
              labels={'index':'Diabates Classification'})
    if variable == "BMI":
        univariate_plot = px.histogram(dataset_copy, x="BMI",title = "BMI Distribution" )
    if variable == "HighBP":
        Hypertension_status = dataset_copy.HighBP.value_counts().reset_index(name = 'Count')
        univariate_plot = px.bar(Hypertension_status, x='index', y = "Count", title = "Hypertension Status", 
              labels={'index':'Hypertension Classification'})
    if variable == "Education": 
        Education = dataset_copy.Education.value_counts().reset_index(name = 'Count')
        univariate_plot = px.bar(Education, x='index', y = "Count", title = "Education Distribution", 
              labels={'index':'Education Category'})
    if variable == "Income": 
        Income = dataset_copy.Income.value_counts().reset_index(name = 'Count')
        univariate_plot = px.bar(Income, x='index', y = "Count", title = "Income Distribution", 
              labels={'index':'Income Category'})
    if variable == "HighChol":
        HighChol = dataset_copy.HighChol.value_counts().reset_index(name = 'Count')
        univariate_plot = px.bar(HighChol, x='index', y = "Count", title = "HighCholDistribution", 
              labels={'index':'HighChol Category'})
    if variable == "Age":
        Age = dataset_copy.Age.value_counts().reset_index(name = 'Count')
        univariate_plot =  px.bar(Age, x='index', y = "Count", title = "Age Distribution", 
              labels={'index':'Age Category'})
    if variable == "PhysActivity": 
        PhysActivity = dataset_copy.PhysActivity.value_counts().reset_index(name = 'Count')
        univariate_plot = px.bar(PhysActivity, x='index', y = "Count", title = "Physical Activity", 
              labels={'index':'Physical Activity Distribution'})
    if variable == "Stroke":
        Stroke = dataset_copy.Stroke.value_counts().reset_index(name = 'Count')
        univariate_plot = px.bar(Stroke, x='index', y = "Count", title = "Stroke Distribution", 
              labels={'index':'Stroke Category'})
    if variable == "HvyAlcoholConsump": 
        HvyAlcoholConsump = dataset_copy.HvyAlcoholConsump.value_counts().reset_index(name = 'Count')
        univariate_plot = px.bar(HvyAlcoholConsump, x='index', y = "Count", title = "HvyAlcoholConsump Distribution", 
              labels={'index':'HvyAlcoholConsump Category'})
    if variable == "GenHlth": 
        GenHlth = dataset_copy.GenHlth.value_counts().reset_index(name = 'Count')
        univariate_plot = px.bar(GenHlth, x='index', y = "Count", title = "General Health Distribution", 
              labels={'index':'General Health Category'})
    if variable == "DiffWalk": 
        DiffWalk = dataset_copy.DiffWalk.value_counts().reset_index(name = 'Count')
        univariate_plot = px.bar(DiffWalk, x='index', y = "Count", title = "Difficulty in walking", 
              labels={'index':'Difficulty in walking Category'})
    if variable == "Sex": 
        Gender = dataset_copy.Sex.value_counts().reset_index(name = 'Count')
        univariate_plot = px.bar(Gender, x='index', y = "Count", title = "Gender", 
              labels={'index':'Gender Distribution'})
    if variable == "Fruits": 
        Fruits = dataset_copy.Fruits.value_counts().reset_index(name = 'Count')
        univariate_plot = px.bar(Fruits, x='index', y = "Count", title = "Fruit Intake", 
              labels={'index':'Fruits Intake Distribution'})
    if variable == "Veggie": 
        Veggies = dataset_copy.Veggies .value_counts().reset_index(name = 'Count')
        univariate_plot = px.bar(Veggies, x='index', y = "Count", title = "Vegetable Intake", 
              labels={'index':'Vegatable Intake Distribution'})
    if variable == "NoDocbcCost": 
        NoDocbcCost = dataset_copy.NoDocbcCost.value_counts().reset_index(name = 'Count')
        univariate_plot = px.bar(NoDocbcCost, x='index', y = "Count", title = "NoDocbcCost Distribution", 
              labels={'index':'NoDocbcCost Category'})
    if variable == "HeartDiseaseorAttack": 
        HeartDiseaseorAttack = dataset_copy.HeartDiseaseorAttack.value_counts().reset_index(name = 'Count')
        univariate_plot = px.bar(HeartDiseaseorAttack, x='index', y = "Count", title = "HeartDiseaseorAttack", 
              labels={'index':'HeartDiseaseorAttack Category'})
    if variable == "MentHlth": 
        univariate_plot = px.box(dataset_copy, y="MentHlth")
    if variable == "PhysHlth": 
        univariate_plot = px.box(dataset_copy, y="PhysHlth")
    if variable == "CholCheck":
        CholCheck = dataset_copy.CholCheck.value_counts().reset_index(name = 'Count')
        univariate_plot = px.bar(CholCheck, x='index', y = "Count", title = "Cholesterol Check Distribution", 
              labels={'index':'Cholesterol Check Category'})
    if variable == "Smoker":
        Smoker = dataset_copy.Smoker.value_counts().reset_index(name = 'Count')
        univariate_plot = px.bar(Smoker, x='index', y = "Count", title = "Smoking Distribution", 
              labels={'index':'Smoking Category'})
    if variable == "AnyHealthcare":
        AnyHealthcare = dataset_copy.AnyHealthcare.value_counts().reset_index(name = 'Count')
        univariate_plot = px.bar(AnyHealthcare, x='index', y = "Count", title = "AnyHealthcare Distribution", 
              labels={'index':'AnyHealthcare Category'})
    
    return univariate_plot

X = data_selected.drop('Diabetes_binary', axis=1)
X = X.values
y = data_selected['Diabetes_binary']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=13)
rf.fit(X_train, y_train)
saved_model = pickle.dumps(rf)
rf_from_pickle = pickle.loads(saved_model)


def diabetes_risk_prediction(HighBP, BMI, PhysHlth, GenHlth, MentHlth,
                             Age, Education, Income, Smoker, Sex):

    indicator_list = [HighBP, BMI, PhysHlth, GenHlth,
                      MentHlth, Age, Education, Income, Smoker, Sex]
    predictions = rf.predict_proba(np.array(indicator_list).reshape(1, -1))
    risk = predictions[0, 1]
    '''
    if risk < 0.3:
        print("You are probably in good health, keep it up.")
    elif risk > 0.7:
        print("See a doctor as soon as you can and listen to their recommendations. You might be on the way to developing diabetes if you don't change your lifestyle.")
        
    elif risk > 0.9:
        print("Go to a hospital right away. Odds are high you have diabetes.")
    else:
        print("You should be alright for the most part, but take care not to let your health slip.")
    '''
    return risk
    # return print("Your Diabetes Risk Index is {:.2f}/50.".format(risk*0.5*100))


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/home')
def data():
    return render_template('home.html')


@app.route('/datares')
def datares():
    return render_template('dataResFairness.html')


@app.route('/aboutdata')
def aboutdata():
    return render_template('aboutdata.html')


@app.route('/datavariables')
def datavariables():
    return render_template('datavariables.html')


@app.route('/background')
def background():
    return render_template('background.html')


@app.route('/summarystatistics')
def summarystatistics():
    return render_template('summarystatistics.html',
                           tables=[summary_cat_transposed.to_html(classes='data')], titles=summary_cat_transposed.columns.values,
                           table_num=[summary_num_transposed.to_html(classes='data')], title_num=summary_num_transposed.columns.values,
                           table_cat_diab=[summary_cat_diabetes_transposed.to_html(classes='data')], title_cat_diab=summary_cat_diabetes_transposed.columns.values,
                           table_num_diab=[summary_num_diabetes_transposed.to_html(classes='data')], title_num_diab=summary_num_diabetes_transposed.columns.values,
                           table_sum_cat_nodiab=[summary_cat_Nodiabetes_transposed.to_html(classes='data')], title_sum_cat_nodiab=summary_cat_Nodiabetes_transposed.columns.values,
                           table_num_nodiab=[summary_num_Nodiabetes_transposed.to_html(classes='data')], title_sum_num_nodiab=summary_num_Nodiabetes_transposed.columns.values,
                           )

@app.route('/univariateanalysis')
def univariateanalysis():
    
    return render_template('univariateanalysis.html')

@app.route('/bivariateanalysis')
def bivariateanalysis():
    bivariate_plot_json=dumps(bivariate_plot,cls=utils.PlotlyJSONEncoder)
    return render_template('bivariateanalysis.html',plot_json=bivariate_plot_json)

@app.route('/bivariateanalysis/<variable>')
def bivariateanalysisgraph(variable):
    bivariate_plot = graphPlot(variable)
    bivariate_plot_json=dumps(bivariate_plot,cls=utils.PlotlyJSONEncoder)
    return render_template('analysisPlot.html',plot_json=bivariate_plot_json)
    ##bivariate_plot_json=dumps(bivariate_plot,cls=utils.PlotlyJSONEncoder)
    #return render_template('bivariateanalysis.html',plot_json=bivariate_plot_json)

@app.route('/univariateanalysis/<variable>')
def univariateanalysisgraph(variable):
    univariate_plot = univariateGraphPlot(variable)
    univariate_plot_json=dumps(univariate_plot,cls=utils.PlotlyJSONEncoder)
    return render_template('analysisPlot.html',plot_json=univariate_plot_json)

@app.route('/correlationanalysis')
def correlationanalysis():
    return render_template('correlationanalysis.html')

@app.route('/predictiveanalysis')
def predictiveanalysis():
    return render_template('predictiveanalysis.html')

@app.route('/riskcalculator', methods=['GET', 'POST'])
def riskcalculator():

    return render_template('riskcalculator.html')


@app.route('/model', methods=['GET', 'POST'])
def model():
    if request.method == "POST":
        HighBP = request.form.get("HighBP")
        BMI = request.form.get("BMI")
        PhysHlth = request.form.get("PhysHlth")
        GenHlth = request.form.get("GenHlth")
        MentHlth = request.form.get("MentHlth")
        Age = request.form.get("Age")
        Education = request.form.get("Education")
        Income = request.form.get("Income")
        Smoker = request.form.get("Smoker")
        Sex = request.form.get("Sex")
        risk = diabetes_risk_prediction(
            HighBP, BMI, PhysHlth, GenHlth, MentHlth, Age, Education, Income, Smoker, Sex)
        if risk < 0.3:
            msg = "Low Risk."
            msg1 = "You are probably in good health, keep it up. Achieve and maintain a healthy body weight be physically active , doing at least 30 minutes of regular, moderate-intensity activity on most days. More activity is required for weight control eat a healthy diet, avoiding sugar and saturated fats avoid tobacco use"
        elif risk > 0.7:
            msg = "Moderate Risk"
            msg1 = "See a doctor as soon as you can and listen to their recommendations. You might be on the way to developing diabetes if you don't change your lifestyle.Achieve and maintain a healthy body weight be physically active , doing at least 30 minutes of regular, moderate-intensity activity on most days. More activity is required for weight control eat a healthy diet, avoiding sugar and saturated fats avoid tobacco use"

        elif risk > 0.9:
            msg = "High Risk."
            msg1 = "Go to a hospital right away. Odds are high you have diabetes. You might be on the way to developing diabetes if you don't change your lifestyle.Achieve and maintain a healthy body weight be physically active , doing at least 30 minutes of regular, moderate-intensity activity on most days. More activity is required for weight control eat a healthy diet, avoiding sugar and saturated fats avoid tobacco use."
        else:
            msg = "Low to Moderate Risk"
            msg1 = "You should be alright for the most part, but take care not to let your health slip.You might be on the way to developing diabetes if you don't change your lifestyle.Achieve and maintain a healthy body weight be physically active , doing at least 30 minutes of regular, moderate-intensity activity on most days. More activity is required for weight control eat a healthy diet, avoiding sugar and saturated fats avoid tobacco use."

        dri = risk*0.5*100

    return render_template('result.html', msg=msg, msg1=msg1, dri=dri)

if __name__ == "__main__":
        app.run(host='127.0.0.1', port=3000, debug=True, use_reloader = True)
