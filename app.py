from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
application=Flask(__name__) #WSGI application

app=application

#route for homepage
@app.route("/") #param specifies the specific url for the web app (in string format), which triggers the index() function automatically
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method=="GET": #user clicked a link to see the form (showing home.html - empty form)
        return render_template("home.html")
    else: #"POST" - user clicked submit button 
        #CustomData(...) takes all values inputted by user to create a CustomData object (with all inputted values as params)
        data=CustomData(
                MWG=float(request.form.get("MWG")), #grabs the value the user typed into the box labelled "MWG" in the HTML
                NWG=float(request.form.get("NWG")),
                KWG=float(request.form.get("KWG")),
                MDIMC=float(request.form.get("MDIMC")),
                NDIMC=float(request.form.get("NDIMC")),
                MDIMA=float(request.form.get("MDIMA")),
                NDIMB=float(request.form.get("NDIMB")),
                KWI=float(request.form.get("KWI")),
                VWM=float(request.form.get("VWM")),
                VWN=float(request.form.get("VWN")),
                STRM=float(request.form.get("STRM")),
                STRN=float(request.form.get("STRN")),
                SA=float(request.form.get("SA")),
                SB=float(request.form.get("SB"))
        )
        pred_df=data.get_data_as_data_frame() #method under CustomData class (creates inputted values to be formatted as a 1-row dataframe)
        print(pred_df)
        predict_pipeline=PredictPipeline() #create a PredictPipeline object
        results=predict_pipeline.predict(pred_df) #find out runtime values using our predict method (input param is going to be our inputted values formatted as a 1-row dataframe)
        return render_template("home.html", results=results[0]) #page is reloaded and results (aka runtimes -> results[0] - numpy array) are now available for the user to see
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True) 