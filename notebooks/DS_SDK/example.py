import new_api
import sdk_wrapper


def get_all_data(measureID):
    # return variable
    all_data = []

    # get all encounters from the api
    encounters = new_api.get_all_encounters()

    for encounter in encounters:
        # Collect the start time, end time and patient_id for each encounter
        start_time_s, end_time_s, patient_id = (
            encounter['start_time'],
            encounter['end_time'],
            encounter['patient_id'],
        )

        # if there was missing data, skip this encounter
        if start_time_s is None or end_time_s is None or patient_id is None:
            continue

        # from the patient_id, get the mrn
        mrn = new_api.patient_id_to_mrn(patient_id)

        # from the mrn and relative time, get the deviceID
        deviceID = new_api.mrn_to_deviceID(mrn, (start_time_s + end_time_s) // 2)

        # If the deviceID wasn't recorded, skip
        if deviceID is None:
            continue

        # convert start and end to milliseconds
        start_time_ms, end_time_ms = int(start_time_s * 1000), int(end_time_s * 1000)

        # Get data from the sdk
        try:
            times, values = sdk_wrapper.get_data(measureID, deviceID, start_time_ms, end_time_ms)
        except:
            continue
        print(times.size)

        # Append new data to result list
        all_data.append([times, values])

    return all_data


if __name__ == "__main__":
    measureID = 157
    all_data = get_all_data(measureID)
