from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import pandas as pd
import os
import json
from .prediction import caller

data = {}
# amenities = ['Resale',  'MaintenanceStaff', 'Gymnasium', 'SwimmingPool', 'LandscapedGardens',
# 'JoggingTrack', 'RainWaterHarvesting', 'IndoorGames', 'ShoppingMall',
# 'Intercom', 'SportsFacility', 'ATM', 'ClubHouse', 'School',
# '24X7Security', 'PowerBackup', 'CarParking', 'StaffQuarter',
# 'Cafeteria', 'MultipurposeRoom', 'Hospital', 'WashingMachine',
# 'Gasconnection', 'AC', 'Wifi', 'Children\'splayarea', 'LiftAvailable',
# 'BED', 'VaastuCompliant', 'Microwave', 'GolfCourse', 'TV',
# 'DiningTable', 'Sofa', 'Wardrobe', 'Refrigerator']

# create api
# will accept iteration
# will return json { "currPrice": XX },

# Create your views here.
def index(request):
    cites = [
        {
            "value" : "Mumbai",
            "text" : "Mumbai"
        },
        {
            "value" : "Delhi",
            "text" : "Delhi",
        },
        {
            "value" : "Chennai",
            "text" : "Chennai",
        },
        {
            "value" : "Hyderabad",
            "text" : "Hyderabad",
        },
        {
            "value" : "Kolkata",
            "text" : "Kolkata",
        },
        {
            "value" : "Bangalore",
            "text" : "Bangalore",
        },


    ]
    data['cities'] = json.dumps(cites)
    return render(request,'chatbot.html',data)

def handle_iteration(request):
    if request.method == 'POST':
        request_data = json.loads(request.POST['data'])
        print(request_data)
        iteration = json.loads(request.POST['iterNo'])
        print('iteration : ',iteration)
        # if(iteration == 1):
        city = request_data.get("city")
        area = request_data.get("area")
        bedroomCount = request_data.get("bedroomCount")
        resale = request_data.get("resale")
        intercom = request_data.get("amenities1").get("intercom")
        security = request_data.get("amenities1").get("security")
        powerBackup = request_data.get("amenities1").get("powerBackup")
        carParking = request_data.get("amenities1").get("carParking")
        lift = request_data.get("amenities1").get("lift")
        maintenanceStaff = request_data.get("amenities2").get("maintenanceStaff")
        gymnasium = request_data.get("amenities2").get("gynasium")
        swimmingPool = request_data.get("amenities2").get("swimmingPool")
        landscape = request_data.get("amenities2").get("landscape")
        jogging = request_data.get("amenities2").get("jogging")
        rainwater = request_data.get("amenities2").get("rainwater")
        sportsFacility = request_data.get("amenities3").get("sportsFacility")
        clubHouse = request_data.get("amenities3").get("clubHouse")
        indoorGames = request_data.get("amenities3").get("indoorGames")
        childrenPlay = request_data.get("amenities3").get("childrenPlay")
        bed = request_data.get("amenities4").get("bed")
        diningTable = request_data.get("amenities4").get("diningTable")
        wardrobe = request_data.get("amenities4").get("wardrobe")
        referigator = request_data.get("amenities4").get("referigator")
        ac = request_data.get("amenities5").get("ac")
        wifi = request_data.get("amenities5").get("wifi")
        tv = request_data.get("amenities5").get("tv")
        microwave = request_data.get("amenities5").get("microwave")

        predicted_price = caller(iteration,city,area,bedroomCount,resale,intercom, security, powerBackup, carParking, lift, maintenanceStaff, gymnasium, swimmingPool, landscape, jogging, rainwater, sportsFacility, clubHouse, indoorGames, childrenPlay, bed, diningTable, wardrobe, referigator, ac, wifi, tv, microwave)

        response = {
            "currPrice" : predicted_price,
        }
        return JsonResponse(response)
