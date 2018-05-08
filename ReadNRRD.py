## changed all Sample to Real (line 10 and line 12) ##
## reads NRRD files and crops cubs as numpy arrays and 2D slices ##
## Tafadzwa Chaunzwa ##

import numpy as np
import pandas as pd
import os
import PreProcessFns as pp
import SimpleITK as sitk

# parameters
DirCuration = '/home/chintan/Desktop/MiniWorkSpace/NRRDs/'
FNChosenSet = 'DeceasedPts.csv'
FNSeedPoint = 'DeceasedPts.csv'
WriteDir = '/home/chintan/Desktop/MiniWorkSpace/'
AxialFD = 'AxialView'
CropFD = 'NPYCrop'
IsotropicFD = 'NPY'
CropSize = 150 # expand from the seed point mm = one voxel

# read the csv file with the manually defined seed points
DFSelection = pd.read_csv(DirCuration + FNChosenSet)
DFSeedPts = pd.read_csv(DirCuration + FNSeedPoint)

FNImages = pd.Series.as_matrix(DFSeedPts.loc[:,'Nrrd'])

# setup dataframe
columns = ['PatientID',
           'NrrdPath', # '' if files not found
           'stackMin','stackMax',
           'orgSpacX','orgSpacY','orgSpacZ',
           'sizeX','sizeY','sizeZ',
           'comX','comY','comZ',
           'NcomX', 'NcomY', 'NcomZ' ,]

ImageParams = pd.DataFrame(columns=columns)


SPs = np.zeros([len(DFSeedPts), 3])
SPs[:, 0] = pd.Series.as_matrix(DFSeedPts.loc[:, 'SPXmm'])
SPs[:, 1] = pd.Series.as_matrix(DFSeedPts.loc[:, 'SPYmm'])
SPs[:, 2] = pd.Series.as_matrix(DFSeedPts.loc[:, 'SPZmm'])

# grab parameters from the image

for i in range(0, len(SPs)):
    #path, file = os.path.split(FNImages[i])
    file = FNImages[i]
    PIN = DFSeedPts['PID'][i]
    #Time = DFSeedPts['StartDate'][i]
    # load nrrd file
    data = sitk.ReadImage(FNImages[i])

    #  MAKE THE IMAGE ISOTROPIC 1mm = 1voxel in all directions
    dataIso = pp.getIsotrpic(data,sitk.sitkLinear)

    # save isptropic image in the same directory as the nrrds
    if not os.path.exists(os.path.join(WriteDir, IsotropicFD, '%04d' % PIN)):
        os.makedirs(os.path.join(WriteDir, IsotropicFD, '%04d' % PIN))
    np.save( os.path.join(WriteDir, IsotropicFD, '%04d' % PIN, '%04d' % PIN + '_'  '.npy') ,
             sitk.GetArrayFromImage(dataIso).astype(np.int16))

    # CROP THE ISO IMAGE
    if not os.path.exists(os.path.join(WriteDir, CropFD, '%04d' % PIN)):
        os.makedirs(os.path.join(WriteDir, CropFD, '%04d' % PIN))

    CropName = os.path.join(WriteDir, CropFD, '%04d' % PIN, 'C%04d' % PIN + '_'  '.npy')
    AxialName = os.path.join(WriteDir, AxialFD, '%04d' % PIN + '_Axial.png')

    SPUnshifted,SPShifted = pp.SaveCrop(dataIso,SPs[i],CropSize,CropName,AxialName)
    print('%04d' % PIN)


    # SAVE PARAMETERS OF EACH IMAGE
    ImageParams.loc[i] = [PIN,
                          DFSeedPts['Nrrd'][i],
                          sitk.GetArrayFromImage(dataIso).min(),
                          sitk.GetArrayFromImage(dataIso).max(),
                          data.GetSpacing()[0], data.GetSpacing()[1], data.GetSpacing()[2],
                          dataIso.GetSize()[0], dataIso.GetSize()[1], dataIso.GetSize()[2],
                          SPUnshifted[0], SPUnshifted[1], SPUnshifted[2],
                          SPShifted[0], SPShifted[1], SPShifted[2]]


ImageParams.to_csv('Params.csv', index = False)
