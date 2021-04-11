# OTUS-FastAPI_Docker_HW

This is FastAPI for the Heart Disease dataset. 
It expects json of the following type:

    {"sample": [63, 1, 3, 145, 233, 2, 0, 80, 0, 2.3, 0, 0, 1]}
or:

    {"sample": "63, 1, 3, 145, 233, 2, 0, 80, 0, 2.3, 0, 0, 1"}
  
POST requests should be send to the following url (in case of local tests):
    
    http://0.0.0.0:5000/heart_post
