import numpy as np
import os
import re

# Before doing any of this you have to manually create a "template" file. I
# did this by opening the spec file I was interested in and then creating a
# new (simplistic) scene and replacing the references to a dscalar file with
# dummary variable names - 'ABSOLUTE_DSCALAR_PATH' & DSCALAR_NAME
# The scene file, by default contains relative paths so I don't think this 
# is easy to transfer to other projects/folders, though the same 
# logic - of using the show-scene command - could be implemented

proj_dir = '/projects/f_mc1689_1/ClinicalActFlow/'

def print_wb_scene(dscalar_filename,output,size='400 250',inflation='standard'):
    '''
    Prints a workbench scene - work in progress
    Inputs:
    dscalar_filename: the dscalar you want to create a png of
    output: output file name (will create a scene and png)
    size: size of png
    '''
    # absolute path to scene template
    if inflation is 'standard':
        scene_template = proj_dir + 'docs/scripts/functions/Template_Amarel.scene'
    elif inflation is 'inflated':
        scene_template = proj_dir + 'docs/scripts/functions/Template_inflated_Amarel.scene'
    elif inflation is 'very':
        scene_template = proj_dir + 'docs/scripts/functions/Template_very_inflated_Amarel.scene'
        
    # create filename labels
    out_scene = output+'.scene'
    out_image = output+'.png'
    
    # open the scene template
    with open(scene_template) as f:
        content = f.readlines()

    # find and replace the dscalar file info
    for i,row in enumerate(content):
        #print(i)
        if re.search('ABSOLUTE_DSCALAR_PATH',row):
            #print('Match found in line',i)
            content[i] = re.sub('ABSOLUTE_DSCALAR_PATH',dscalar_filename,row)
        elif re.search('DSCALAR_NAME',row):
            #print('Match found in line',i)
            content[i] = re.sub('DSCALAR_NAME',dscalar_filename,row)

    # write the changed file out
    #print('creating scene')
    if os.path.isfile(out_scene):
        os.remove(out_scene)
    with open(out_scene,'w') as f:
        for i,row in enumerate(content):
            f.write(content[i])

    # use connectome workbench to print the scene
    #print('converting scene using wb')
    if os.path.isfile(out_image):
        os.remove(out_image)
    cmd = 'wb_command -show-scene '+out_scene+' 1 '+out_image+' '+size
    #print(cmd)
    os.system(cmd)