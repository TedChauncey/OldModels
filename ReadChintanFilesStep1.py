import csv
import SimpleITK as sitk
import numpy
import os

#f = open('manual_segs.csv')
f = open('LIDC_Master.csv')
csv_f = csv.reader(f)
next(csv_f)

ID = []
nod = []
read = []
mal = []
centroidzyx = []

Size = 25 # set crop size = Size.2 = 50

for row in csv_f:
    ID.append(row[0])
    nod.append(row[1])
    read.append(row[2])
    centroidzyx.append((row[4],row[5],row[6]))
f.close()

for i in range(len(ID)):
    img = sitk.ReadImage(ID[i] +"/UnknownStudy/Reconstructions/image.nii.gz")
    data = sitk.GetArrayFromImage( img )
    
    img_seg = sitk.ReadImage(ID[i] +"/UnknownStudy/Segmentations/read_" + read[i] + "_manual_nodule_" + nod[i] + ".nii.gz")
    data_seg = sitk.GetArrayFromImage( img_seg )
    
    x = int(centroidzyx[i][0])
    y = int(centroidzyx[i][1])
    z = int(centroidzyx[i][2])
    
    zmin = z-Size
    zmax = z+Size
    xmin = x-Size
    xmax = x+Size
    ymin = y-Size
    ymax = y+Size
    cx = Size
    cy = Size
    cz = Size
    
    #check bounds
    if (x < Size):
        cx = x
        xmin = 0
        xmax = Size*2
    if ((x + Size) > data.shape[2]):
        cx = Size*2 - (data.shape[2] - x)
        xmin = data.shape[2] - Size*2
        xmax = data.shape[2]
        
    if (y < Size):
        cy = y
        ymin = 0
        ymax = Size*2
    if ((y + Size) > data.shape[1]):
        cy = 2*Size - (data.shape[1] - y)
        ymin = data.shape[1] - Size*2
        ymax = data.shape[1]
    
    if (z < Size):
        cz = z
        zmin = 0
        zmax = Size*2
    if ((z + Size) > data.shape[0]):
        cz = Size*2 - (data.shape[0] - z)
        zmin = data.shape[0] - Size*2
        zmax = data.shape[0]
    
    nodule_patch = data[zmin:zmax,ymin:ymax,xmin:xmax]
    #print nodule_patch.shape

    nodule_cube = sitk.GetImageFromArray(nodule_patch)
    nodule_cube.SetSpacing([1.0,1.0,1.0])
    
    nodule_patch = sitk.GetArrayFromImage(nodule_cube)
    print nodule_patch.shape
    print nodule_patch

    sitk.WriteImage(nodule_cube, "resampled_cubes/" + ID[i] + "_" + read[i] + "_" + nod[i] +".nrrd") ## save funky file formats as nrrds
    
    """flattened_nodule_patch = nodule_patch[cz,:,:]
    
    nodule_slice = sitk.GetImageFromArray (flattened_nodule_patch)
        
    sitk.WriteImage(nodule_slice, "sliced_images_xy_raw_resampled/" + ID[i] + "_" + read[i] + "_" + nod[i] +".nii.gz")
    
    flattened_nodule_patch_2 = nodule_patch[:, cy, :]

    nodule_slice_2 = sitk.GetImageFromArray(flattened_nodule_patch_2)
        
    sitk.WriteImage(nodule_slice_2, "sliced_images_xz_raw_resampled/" + ID[i] + "_" + read[i] + "_" + nod[i] +".nii.gz")
    
    flattened_nodule_patch_3 = nodule_patch[:, :, cx]

    nodule_slice_3 = sitk.GetImageFromArray (flattened_nodule_patch_3)
        
    sitk.WriteImage(nodule_slice_3, "sliced_images_yz_raw_resampled/" + ID[i] + "_" + read[i] + "_" + nod[i] +".nii.gz")"""

    
    
    
    
    
    
    
    
    
    
    
    
    
