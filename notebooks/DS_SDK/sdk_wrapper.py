from ctypes import *
import os
import numpy as np

PATH = os.path.realpath(__file__)
PATH = PATH.split("sdk_wrapper")[0]
sdk_dll = CDLL(PATH + "sdk_lite.so")

sdk_dll.load_block_index.argtypes = [c_int, c_char_p, c_char_p]
sdk_dll.load_block_index.restype = c_uint

sdk_dll.set_q_size.argtypes = [c_uint]

sdk_dll.get_data.argtypes = [
    POINTER(c_ulonglong),
    POINTER(c_longlong),
    c_ulonglong,
    c_ulonglong,
    c_ulonglong,
    POINTER(c_ulonglong),
    POINTER(c_ulonglong),
    c_int,
    c_char_p,
    c_char_p,
    c_char_p,
]

sdk_dll.get_data_init.argtypes = [c_ulonglong, c_ulonglong, c_int, c_char_p, c_char_p, c_char_p]
sdk_dll.get_data_init.restype = c_ulonglong


def main():
    measureID, deviceID = 157, "3-109_1"
    load_block_index(measureID, deviceID)
    times, values = get_data(measureID, deviceID, 1540307115 * 1000, (1550307115 * 1000) + 10000000)
    print(times, values)
    print(times.size)


def load_block_index(measureID, deviceID, db_name="atrium_db_1", top_lvl_dir="/mnt/datasets/Will_SDK/"):
    deviceID = bytes(deviceID, 'utf-8')
    db_name = bytes(db_name, 'utf-8')
    top_lvl_dir = bytes(top_lvl_dir, 'utf-8')
    sdk_dll.load_block_index(measureID, deviceID, db_name, top_lvl_dir)


def set_buffer_size(size):
    sdk_dll.set_q_size(size)


def get_data(
    measureID, deviceID, start_time, end_time, db_name="atrium_db_1", top_lvl_dir="/mnt/datasets/Will_SDK/"
):
    deviceID = bytes(deviceID, 'utf-8')
    db_name = bytes(db_name, 'utf-8')
    top_lvl_dir = bytes(top_lvl_dir, 'utf-8')

    num_vals = sdk_dll.get_data_init(start_time, end_time, measureID, deviceID, db_name, top_lvl_dir)

    left, right = c_ulonglong(), c_ulonglong()
    times, values = np.zeros(num_vals, dtype=np.uint64), np.zeros(num_vals, dtype=np.int64)
    sdk_dll.get_data(
        times.ctypes.data_as(POINTER(c_ulonglong)),
        values.ctypes.data_as(POINTER(c_longlong)),
        num_vals,
        start_time,
        end_time,
        byref(left),
        byref(right),
        measureID,
        deviceID,
        db_name,
        top_lvl_dir,
    )
    left, right = left.value, right.value
    # print(times, values)
    return values[left:right].astype(np.int32)


set_buffer_size(1)

if __name__ == "__main__":
    main()
