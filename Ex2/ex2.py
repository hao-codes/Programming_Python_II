"""
Author: Hao Zheng

Exercise 2
"""


"""
-get list of files in input_dir
-sort list of files
-process files if they are valid and then copy them to output_dir

-If output_dir or the directory containing log_file does not exist, your function should create them.

files are valid if:
1. file name ends with .jpg, .JPG, .jpeg or .JPEG.
2. file size max 250 000 bytes
3. file can be read as image - PIL/ pillow doesnt raise an exception
4. image data has shape of (H, W, 3): Height, WIdth are 96 pix or larger
color channel must be in RGB order
5. image data has variance > 0 --> more than only one common pixel color in the img
6. the same image has not been copied already


copied files:
basename defined by formatter:
starts with 0, +1 for every copied file
files must end with .jpg

names of invalid files should be written to log_file:
each line contains the file name of the invalid file, followed by semicolon, an error code and newline character
error code is int  with 1 digit - corresponding to above 6 rules
file name should only contain relative file paths

function should return number of valid files that were copied

"""
import numpy as np
import os
import PIL
import hashlib
import shutil
from PIL import Image
import glob
from PIL.ImageStat import Stat



def validate_images(input_dir: str, output_dir: str, log_file: str, formatter=""):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    error_code = 0
    img_hashes_list = []
    n_valid_files = 0
    files = glob.glob(os.path.join(input_dir, '**', '*'), recursive=True)
    files.sort()

    files = [file for file in files if os.path.isdir(file) == False]

    for input_file in files:
        # print(input_file)

        if not (input_file.endswith(".jpg") or input_file.endswith(".JPG") or input_file.endswith(
                ".jpeg") or input_file.endswith(".JPEG")):
            error_code = 1
            error_message = input_file.split("/")[-1] + ";" + str(error_code) + "\n"
            with open(log_file, "a") as logs:
                logs.write(error_message)
            logs.close()
            continue

        if not os.path.getsize(input_file) < 250000:
            error_code = 2
            error_message = input_file.split("/")[-1] + ";" + str(error_code) + "\n"
            with open(log_file, "a") as logs:
                logs.write(error_message)
            logs.close()
            continue

        try:
            img = Image.open(input_file)
        except:
            error_code = 3
            error_message = input_file.split("/")[-1] + ";" + str(error_code) + "\n"
            with open(log_file, "a") as logs:
                logs.write(error_message)
            logs.close()
            continue

        width, height = img.size
        mode = img.mode
        img_array = np.array(img)
        if height < 96 or width < 96 or mode != "RGB":
            error_code = 4
            error_message = input_file.split("/")[-1] + ";" + str(error_code) + "\n"
            with open(log_file, "a") as logs:
                logs.write(error_message)
            logs.close()
            continue
        # if Stat(img).var == 0:
        if not np.var(img_array, 1).all() > 0.0:
            error_code = 5
            error_message = input_file.split("/")[-1] + ";" + str(error_code) + "\n"
            with open(log_file, "a") as logs:
                logs.write(error_message)
            logs.close()
            continue

        img_bytes = img_array.tostring()
        hash_function = hashlib.sha256()
        hash_function.update(img_bytes)

        img_hashed = hash_function.digest()

        if img_hashed in img_hashes_list:
            error_code = 6
            error_message = input_file.split("/")[-1] + ";" + str(error_code) + "\n"
            with open(log_file, "a") as logs:
                logs.write(error_message)
            logs.close()
            continue
        # add img_hash to list
        # copy img to output dir with correct name & format
        # update copied images counter
        img_hashes_list.append(img_hashed)

        copied_name = ("{:" + formatter + "}").format(n_valid_files) + ".jpg"
        shutil.copy(input_file, os.path.join(output_dir, copied_name))
        n_valid_files += 1

    return n_valid_files




def getAllFiles(input_dir):
    input_files_list = []
    listOfFiles = os.listdir(input_dir)
    for entry in listOfFiles:
        fullpath = os.path.join(input_dir, entry)
        if os.path.isdir(fullpath):
            input_files_list = listOfFiles + getAllFiles(fullpath)
        else:
            input_files_list.append(fullpath)
    return input_files_list


