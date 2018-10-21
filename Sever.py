import  pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, flash, redirect, render_template, request, session, abort
import json


app=Flask(__name__)

data=pd.read_csv('national_universities_rankings.csv')
data.at[4,'name']='Columbia University in the City of New York'

data['in_state']=['Private' if pd.isnull(i) else 'Public' for i in data['in_state']]
tfidf = TfidfVectorizer(stop_words='english')
data['description']=data['description'].fillna('')
tfidf_matrix= tfidf.fit_transform(data['description'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(data.index, index=data['name']).drop_duplicates()

def recommendation(name,cosine_sim=cosine_sim):
    idx = indices[name]
    sim_scores=list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    school_indices = [i[0] for i in sim_scores]
    return data['name'].iloc[school_indices].tolist()



data2=pd.read_csv('Most-Recent-Cohorts-Scorecard-Elements.csv')
data2=data2.rename(index=str,columns={'INSTNM':'name'})
data2['SAT']=data2['SATMT25']+data2['SATVR25']
data3=pd.read_csv("MERGED2015_16_PP.csv")


data3=data3.rename(index=str,columns={'INSTNM':'name'})
data3=data3['ICLEVEL']
data3=data3.tolist()

data2['level']=data3
data2=data2[data2['level']==1]

data2['SAT score']=data2['SAT'].fillna(1000)
data2['% Comp Sci']=data2['PCIP11'].fillna(0)
data2['% Engineering']=data2['PCIP14'].fillna(0)
data2['% Math']=data2['PCIP27'].fillna(0)
data2['%Social Sciences']=data2['PCIP45'].fillna(0)
data2['%English and Literature']=data2['PCIP23'].fillna(0)
data2['%Biology']=data2['PCIP26'].fillna(0)
data2['%Business']=data2['PCIP52'].fillna(0)

df=data2[['name','CITY','LOCALE','INSTURL','RET_FT4','SAT score','% Comp Sci',
       '% Engineering','% Math','%Social Sciences','%English and Literature','%Biology','%Business']]

df = df.reset_index(drop=True)
df['retention rate']=df['RET_FT4'].fillna(0)

df['LOCALE']=[
'city' if i==11.0 or i == 12.0 else 'suburban' if 12<i and i<21 else 'rural' for i in df['LOCALE']
    ]


@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/recommend/")
def schol_rec():
    return render_template('recommend.html')

@app.route("/recommend/result",methods=['GET','POST'])
def school_result():
    school=request.form['text']
    school=data.loc[data['name'].isin(recommendation(school))]
    # school=school[['name','CITY','INSTURL','SAT score']]
    # school.columns=['University','Location','Website','Average SAT Score']
    school=school.drop(columns=['description'])
    school=school.reset_index()
    school=school.drop(columns=['index'])
    return school.to_html()


@app.route("/QA/")
def question():
    return render_template('QA.html')

@app.route("/QA/result",methods=['GET','POST'])
def result():
    location=request.form['location']
    major=request.form['major']
    sat=request.form['sat']

    if sat == '1400-1600':
        sat_threshold_up = 1600
        sat_threshold_low = 1299
    elif sat == '1200-1400':
        sat_threshold_up = 1100
        sat_threshold_low = 900
    else:
        sat_threshold_up = 900
        sat_threshold_low = 0
    df_loc=df[(df.LOCALE)==location]
    df_sat=df_loc[(df_loc['SAT score']>= sat_threshold_low)&(df_loc['SAT score']<=sat_threshold_up)]
    
    if major == 'math':
        df_math=df_sat.sort_values(by=['% Math'],ascending=False)
        result=df_math[['name','CITY','INSTURL','retention rate']].head(10)
        result.columns=['School name', 'Location','School Website','retention rate']
    
    elif major == 'cs':
        df_cs=df_sat.sort_values(by=['% Comp Sci'],ascending=False)
        result=df_cs[['name','CITY','INSTURL','retention rate']].head(10)
        result.columns=['School name', 'Location','School Website','retention rate']

    elif major == 'engin':
        df_engin=df_sat.sort_values(by=['% Engineering'],ascending=False)
        result=df_engin[['name','CITY','INSTURL','retention rate']].head(10)
        result.columns=['School name', 'Location','School Website','retention rate']

    elif major == 'ss':
        df_ss=df_sat.sort_values(by=['%Social Sciences'],ascending=False)
        result=df_ss[['name','CITY','INSTURL','retention rate']].head(10)
        result.columns=['School name', 'Location','School Website','retention rate']

    elif major =='el':
        df_el=df_sat.sort_values(by=['%English and Literature'],ascending=False)
        result=df_el[['name','CITY','INSTURL','retention rate']].head(10)
        result.columns=['School name', 'Location','School Website','retention rate']

    elif major == 'buss':
        df_buss=df_sat.sort_values(by=['%Business'],ascending=False)
        result=df_buss[['name','CITY','INSTURL','retention rate']].head(10)
        result.columns=['School name', 'Location','School Website','retention rate']

    elif major == 'bio':
        df_bio=df_sat.sort_values(by=['%Business'],ascending=False)
        result=df_bio[['name','CITY','INSTURL','retention rate']].head(10)
        result.columns=['School name', 'Location','School Website','retention rate']
       
    result=result.reset_index()
    result=result.drop(columns=['index'])
    return result.to_html()















'''
@app.route("/QA/result_test",methods=['GET','POST'])
def result_test():
    location=request.form['location']
    #return location
    major=request.form['major']
    sat=request.form['sat']
    if (location=='city' and sat=='1400-1600' and major=='math'):
        df_loc=df[(df.LOCALE==11) | (df.LOCALE==12)]
        df_sat=df_loc[df_loc['SAT score']>1300]
        df_math=df_sat.sort_values(by=['% Math'],ascending=False)
        result=df_math[['name','CITY','INSTURL']].head(10)
        result.columns=['School name', 'Location','School Website']
    elif (location=='suburban' and sat=='1400-1600' and major=='math'):
        df_loc=df[(12<df.LOCALE) & (df.LOCALE<=21)]
        df_sat=df_loc[df_loc['SAT score']>1300]
        df_math=df_sat.sort_values(by=['% Math'],ascending=False)
        result=df_math[['name','CITY','INSTURL']].head(10)
        result.columns=['School name', 'Location','School Website']
    elif (location=='rural' and sat=='1400-1600' and major=='math'):
        df_loc=df[df.LOCALE>21]
        df_sat=df_loc[df_loc['SAT score']>1300]
        df_math=df_sat.sort_values(by=['% Math'],ascending=False)
        result=df_math[['name','CITY','INSTURL']].head(10)
        result.columns=['School name', 'Location','School Website']
    elif (location=='city' and sat=='1400-1600' and major=='cs'):
        df_loc=df[(df.LOCALE==11) | (df.LOCALE==12)]
        df_sat=df_loc[df_loc['SAT score']>1300]
        df_math=df_sat.sort_values(by=['% Comp Sci'],ascending=False)
        result=df_math[['name','CITY','INSTURL']].head(10)
        result.columns=['School name', 'Location','School Website']
    elif (location=='suburban' and sat=='1400-1600' and major=='cs'):
        df_loc=df[(12<df.LOCALE) & (df.LOCALE<=21)]
        df_sat=df_loc[df_loc['SAT score']>1300]
        df_math=df_sat.sort_values(by=['% Comp Sci'],ascending=False)
        result=df_math[['name','CITY','INSTURL']].head(10)
        result.columns=['School name', 'Location','School Website']
    elif (location=='rural' and sat=='1400-1600' and major=='cs'):
        df_loc=df[df.LOCALE>21]
        df_sat=df_loc[df_loc['SAT score']>1300]
        df_math=df_sat.sort_values(by=['% Comp Sci'],ascending=False)
        result=df_math[['name','CITY','INSTURL']].head(10)
        result.columns=['School name', 'Location','School Website']          
    return result.to_html()
'''

if __name__ == "__main__":
	app.run(debug=True)
            
        
            
            
 





