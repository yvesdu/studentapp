from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/getperformance',methods=['POST','GET'])
def get_delay():
    if request.method=='POST':
        result=request.form

        #Prepare the feature vector for prediction
        pkl_file = open('cat', 'rb')
        index_dict = pickle.load(pkl_file)
        new_vector = np.zeros(len(index_dict))

        try:
            new_vector[index_dict['gender_'+str(result['gender'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['NationalITy_'+str(result['NationalITy'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['PlaceofBirth_'+str(result['PlaceofBirth'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['StageID_'+str(result['StageID'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['GradeID_'+str(result['GradeID'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['SectionID_'+str(result['SectionID'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['Topic_'+str(result['Topic'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['Semester_'+str(result['Semester'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['Relation_'+str(result['Relation'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['ParentAnsweringSurvey_'+str(result['ParentAnsweringSurvey'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['ParentschoolSatisfaction_'+str(result['ParentschoolSatisfaction'])]] = 1
        except:
            pass
        try:
            new_vector[index_dict['StudentAbsenceDays_'+str(result['StudentAbsenceDays'])]] = 1
        except:
            pass


        pkl_file = open('model5.pkl', 'rb')
        model5 = pickle.load(pkl_file)
        labels5 = model5.predict(new_vector)

        return render_template('result.html',labels5=labels5)


if __name__ == '__main__':
    app.run()
