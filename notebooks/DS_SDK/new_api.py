import requests
import json
import csv

API_URL = "http://lore-tsc.sickkids.ca:5000/"
USERNAME = 'dmitrii'
PASSWORD = '4CKNrB^@({Qtzz@n'

# names = ['53-3', '53-4', '53-5', '53-6', '53-7', '53-8', '59', '60', '63', '64', '73-1', '73-2', '73-3', '73-4', '75-1', '75-2', '75-3', '75-4', '76-1', '76-2', '76-3', '76-4', '80-1', '80-2', '80-3', '80-4', '84', '88', '93-1', '93-2', '93-3', '93-4', '94-1', '94-2', '94-3', '94-4', '107', '109', '110', '111', '112', '113', '114', '115', '116']
# dev_ids = ['3-53_3', '3-53_4', '3-53_5', '3-53_6', '3-53_7', '3-53_8', '3-59_1', '3-60_1', '3-63_1', '3-64_1', '3-73_1', '3-73_2', '3-73_3', '3-73_4', '3-75_1', '3-75_2', '3-75_3', '3-75_4', '3-76_1', '3-76_2', '3-76_3', '3-76_4', '3-80_1', '3-80_2', '3-80_3', '3-80_4', '3-84_1', '3-88_1', '3-93_1', '3-93_2', '3-93_3', '3-93_4', '3-94_1', '3-94_2', '3-94_3', '3-94_4', '3-107_1', '3-109_1', '3-110_1', '3-111_1', '3-112_1', '3-113_1', '3-114_1', '3-115_1', '3-116_1']
"""
name_to_dev_id = {}

for i in range(len(names)):
    name_to_dev_id[names[i]] = dev_ids[i]
"""


def main():
    device_dict = get_device_dict()
    result = []
    with open("artefact_summary.csv") as file_in, open('result_summary.csv', 'w+') as file_out:
        reader = csv.reader(file_in)
        writer = csv.writer(file_out)
        data = list(reader)

        for i, row in enumerate(data):
            if i == 0:
                result.append(row + ["age", "gender"])
                continue
            deviceID, start_time, end_time = row[1], row[5], row[6]
            bed_id = device_dict[deviceID]
            encounter = api_get(
                'encounters',
                payload={'bed_id': int(bed_id), 'start_time': int(start_time), 'end_time': int(end_time)},
            )
            patient_id = encounter[0]['patient_id']
            patient_info = api_get('patients/{}'.format(patient_id))
            age, gender = int(start_time) - patient_info['dob'], patient_info['gender']
            result.append(row + [age, gender])

        writer.writerows(result)


def login(username, password):
    response = requests.post(API_URL + 'auth/login', json={'username': username, 'password': password})
    if response.status_code != 200:
        return None
    else:
        return response.json()["access_token"]


ACCESS_TOKEN = login(USERNAME, PASSWORD)
if ACCESS_TOKEN is None:
    print('Failed to Authenticate')
    exit()


def device_cache():
    devices = api_get('devices')
    cache = {}
    for device in devices:
        cache[device["id"]] = device["name"]

    return cache


def mrn_to_deviceID(mrn, time_s):
    global ACCESS_TOKEN
    encounters = get_encounters(mrn, time_s, time_s)
    # print(encounters)
    if len(encounters) == 0:
        return None
    counter = 0
    while counter < 10 and isinstance(encounters, dict):
        ACCESS_TOKEN = login(USERNAME, PASSWORD)
        if ACCESS_TOKEN is None:
            print('Failed to Authenticate')
            exit()
        encounters = get_encounters(mrn, time_s, time_s)
        counter += 1

    encounter_id = encounters[0]["id"]
    device_list = get_devices(encounter_id)
    if len(device_list) == 0:
        return None
    new_deviceID = device_list[0]['device_id']
    dev_id = get_real_device(new_deviceID)
    return dev_id['name']


def patient_id_to_mrn(patient_id):
    response = api_get('patients/{}'.format(patient_id))
    return response['mrn']


def get_all_encounters(mrn=None):
    if mrn is not None:
        return api_get('encounters', payload={'mrn': str(mrn)})
    return api_get('encounters')


def get_encounters(mrn, start_time, end_time):
    return api_get(
        'encounters', payload={'mrn': str(mrn), 'start_time': str(start_time), 'end_time': str(end_time)}
    )


def get_devices(encounter_id):
    return api_get('encounters/{}/devices'.format(encounter_id))


def api_get(endpoint, payload=None):
    headers = {'Authorization': 'Bearer ' + ACCESS_TOKEN}
    response = requests.get('{}/api/'.format(API_URL) + endpoint, params=payload, headers=headers)
    return response.json()


def get_device_dict():
    result = {}
    bed_to_vendor_name_dict = {
        '107': '3-107_1',
        '109': '3-109_1',
        '110': '3-110_1',
        '111': '3-111_1',
        '75-1': '3-75_1',
        '75-2': '3-75_2',
    }
    device_list = api_get('beds')

    for device_info in device_list:
        # device = api_get('devices/{}'.format(device_info['id']))
        if device_info['name'] not in bed_to_vendor_name_dict:
            continue
        result[bed_to_vendor_name_dict[device_info['name']]] = device_info['id']
    return result


def get_real_device(device_id):
    return api_get('devices/{}'.format(device_id))


# Run the example
if __name__ == '__main__':
    main()
