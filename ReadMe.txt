RiffleShuffle
A Supervised, Symmetry-Driven, GUI Toolkit for Mouse Brain Stack Registration and Plane Assignment


Pipeline:

rsPreProcessing
rsStackRegistration
rsPlaneAssignment


Auxiliary scripts:

rsTrainContour
rsTrainMask


The code is designed to make use of Matlab's 'code cells',
where a cell is a section of code delimited by lines starting with '%%',
which can be executed using CTRL+Enter.


Parameters that need to be set are indicated like so:
%\\\SET
    parameter = x;
%///


Developed by Marcelo Cicconet & Daniel R. Hochbaum

-----

Some sample data is available for demo purposes at
https://www.dropbox.com/s/gs5hb2w187vcp21/RiffleShuffle.zip?dl=0

It contains 3 folders:

Stacks
    Here, subfolder Dataset_B_1.55_2.45_Downsized contains the output
    of step 1 (rsPreProcessing), which serves as input of step 2 (rsStackRegistration).
    The original data is not released due to the file size being very large.
    Furthermore, it's very likely that the user's preprocessing will be different.
    
SupportFiles
    Contains Atlas data used in rsPlaneAssignment.
    This was obtained form https://mouse.brain-map.org/

DataForPC
    Sample data for rsTrainContour and rsTrainMask.
    If the user wants to train machine learning models to generate contour likelihoods
    and slice masks for step 1 (rsPreProcessing), this data can be used in conjunction
    with rsTrainContour and rsTrainMask as a guideline.

-----

Reference:
    M. Cicconet and D. R. Hochbaum.
    Interactive, Symmetry Guided Registration to the Allen Mouse Brain Atlas.
    [To appear at] BioImage Informatics 2019. Seattle, WA.

-----

For questions, contact marcelo_cicconet [at] hms [dot] harvard [dot] edu