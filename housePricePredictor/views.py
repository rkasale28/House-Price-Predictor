from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd

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
    data['amenities'] = amenities
    return render(request,'index.html',data)

def result(request):
    if request.method=='POST':
        data['amenities'] = amenities
        data['city'] = request.POST['cities']
        data['area'] = request.POST['area']
        data['bedrooms'] = request.POST['bedrooms']

        dict = {}
        for i in amenities:
            dict[i] = request.POST.__contains__(i.lower())
        data['dict'] = dict
        
        return render(request,'result.html',data)

    return HttpResponse('Not Found')
