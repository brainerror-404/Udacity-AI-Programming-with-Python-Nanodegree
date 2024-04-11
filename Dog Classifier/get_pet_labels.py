#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#                                                                             
# PROGRAMMER: Ooi Shu Hui
# DATE CREATED: 29/6/23
# REVISED DATE: 
# PURPOSE: Create the function get_pet_labels that creates the pet labels from 
#          the image's filename. This function inputs: 
#           - The Image Folder as image_dir within get_pet_labels function and 
#             as in_arg.dir for the function call within the main function. 
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main. 
#          The results_dic dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).
#
##
# Imports python modules
from os import listdir

# TODO 2: Define get_pet_labels function below please be certain to replace None
#       in the return statement with results_dic dictionary that you create 
#       with this function
# 
def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels (results_dic) based upon the filenames 
    of the image files. These pet image labels are used to check the accuracy 
    of the labels that are returned by the classifier function, since the 
    filenames of the images contain the true identity of the pet in the image.
    Be sure to format the pet labels so that they are in all lower case letters
    and with leading and trailing whitespace characters stripped from them.
    (ex. filename = 'Boston_terrier_02259.jpg' Pet label = 'boston terrier')
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by the classifier function (string)
    Returns:
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
      List. The list contains for following item:
         index 0 = pet image label (string)
    """
    # Replace None with the results_dic dictionary that you created with this
    # function
    # Creates list of files in directory
    filename = listdir(image_dir)

    # Creates empty dictionary for the results
    results_dic = dict()

    # Processes through each file in the directory, extracting only the words
    # of the file that contain the pet image label
    for i in range(0, len(filename), 1):
        # Skips file if starts with . (like .DS_Store of Mac OSX) because it
        # isn't a pet image file
        if filename[i][0] != ".":
            # Create a temporary pet label variable
            pet_label = ""
            pet_name = filename[i].lower().split("_")

            # Check if all the names are alphabetic
            for name in pet_name:
                if name.isalpha():
                    pet_label += name + " "
            pet_label = pet_label.strip()

        # If filename doesn't already exist in dictionary add it, and it's
        # pet label - otherwise print an error message because indicates
        # duplicate files (filenames)
        if filename[i] not in results_dic:
            results_dic[filename[i]] = [pet_label]
        else:
            print("Duplicated files detected!")
            print(filename[i] + " already existed in dictionary.")

    # print("Filename and pet labels in the dictionary:")
    # for key in results_dic:
    #     print("Filename: " + key + "    Pet label: " + results_dic[key][0])

    return results_dic
