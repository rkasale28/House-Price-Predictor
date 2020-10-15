from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import os

data = {}
amenities = ['Resale',  'MaintenanceStaff', 'Gymnasium', 'SwimmingPool', 'LandscapedGardens',
'JoggingTrack', 'RainWaterHarvesting', 'IndoorGames', 'ShoppingMall',
'Intercom', 'SportsFacility', 'ATM', 'ClubHouse', 'School',
'24X7Security', 'PowerBackup', 'CarParking', 'StaffQuarter',
'Cafeteria', 'MultipurposeRoom', 'Hospital', 'WashingMachine',
'Gasconnection', 'AC', 'Wifi', 'Children\'splayarea', 'LiftAvailable',
'BED', 'VaastuCompliant', 'Microwave', 'GolfCourse', 'TV',
'DiningTable', 'Sofa', 'Wardrobe', 'Refrigerator']


# Create your views here.
def index(request):
    data['cities'] = ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai']
    return render(request,'index.html',data)

def intermediate(request):
    data['amenities'] = amenities
    print (len(amenities))
    if request.method=='POST':
        city = request.POST['cities']

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df = pd.read_csv(os.path.join(BASE_DIR, 'predictor/data',city+'.csv'))
        data['locations'] = df['Location'].unique()
        data['city'] = city

        return render(request,'intermediate.html',data)

    return HttpResponse('404 : Not Found')

def result(request):
    if request.method=='POST':
        data['amenities'] = amenities
        data['city'] = request.POST['city']
        data['area'] = request.POST['area']
        data['bedrooms'] = request.POST['bedrooms']
        data['location'] = request.POST['locations']

        dict = {}
        for i in amenities:
            if request.POST.__contains__(i.lower()):
                dict[i] = 'Yes'
            else:
                dict[i] = 'No'    
        data['dict'] = dict

        return render(request,'result.html',data)

    return HttpResponse('404 : Not Found')
