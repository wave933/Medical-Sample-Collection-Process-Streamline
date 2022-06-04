
    
import pickle
import pandas as pd
from flask import Flask, render_template, request

# app instance
app = Flask(__name__)

# loading dataset and model
df = pd.read_csv(open('Cleaned_data.csv','rb'))
#df = pd.read_excel(r"C:\Users\Shreni Singh\Desktop\medical_sample_streamline\sample_data.xlsx")
rf_model = pickle.load(open('./rf_cl_model.pkl', 'rb'))


# index page


@app.route('/', methods=['GET', 'POST'])
def index():
    test_name = sorted(df['Test_Name'].unique())
    # sample name dont need to be here used radio buttons insted
    # also way of storage same
    # same for cut off schedule
    agent_id = sorted(df ['Agent_ID'].unique())
    # again same for all others
    return render_template('index.html', test_name=test_name)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    test_name = request.form.get('testname')
    samples = request.form.get('exampleRadios')
    storage_type = request.form.get('newradio')
    cut_off_schedule = request.form.get('secondradio')
    cut_off_time = request.form.get('cutofftime')
    traffic_condition = request.form.get('traffic')
    agents_location = request.form.get('agentlocation')
    t_to_r_patient = request.form.get('reachpatienttime')
    t_for_s_location = request.form.get('collectiontime')
    lab_location = request.form.get('lablocation')
    t_taken_to_reach_lab = request.form.get('timeoflab')
    patient_age = request.form.get('patientAge')

    data = {
        'Test_Name': test_name,
        'Sample': samples,
        'Way_Of_Storage_Of_Sample': storage_type,
        'Cut_off_Schedule': cut_off_schedule,
        'Cut_off_time_HH_MM': cut_off_time,
        'Traffic_Conditions': traffic_condition,
        'Agent_Location_KM': agents_location,
        'Time_Taken_To_Reach_Patient_MM':t_to_r_patient,
        'Time_For_Sample_Collection_MM': t_for_s_location,
        'Lab_Location_KM': lab_location,
        'Time_Taken_To_Reach_Lab_MM': t_taken_to_reach_lab,
        'Patient_Age': patient_age
    }

    features = pd.DataFrame(data, index=[0])
    pred = rf_model.predict(features)

    if pred == 'Y':
        return render_template("predict.html", prediction='YES')
    else:
        return render_template("predict.html", prediction='NO')


if __name__ == "__main__":
    app.run(debug=True)