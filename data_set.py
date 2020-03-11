# -*- coding: utf-8 -*-

from collections import namedtuple

Data_set = namedtuple('Data_set',['name','classes_num','train_size','test_size','length'])

ACSF1 = Data_set(name='ACSF1', classes_num=10, train_size=100, test_size=100, length=1460)
Adiac = Data_set(name='Adiac', classes_num=37, train_size=390, test_size=391, length=176)
AllGestureWiimoteX = Data_set(name='AllGestureWiimoteX', classes_num=10, train_size=300, test_size=700, length=500)
AllGestureWiimoteY = Data_set(name='AllGestureWiimoteY', classes_num=10, train_size=300, test_size=700, length=500)
AllGestureWiimoteZ = Data_set(name='AllGestureWiimoteZ', classes_num=10, train_size=300, test_size=700, length=500)
ArrowHead = Data_set(name='ArrowHead', classes_num=3, train_size=36, test_size=175, length=251)
Beef = Data_set(name='Beef', classes_num=5, train_size=30, test_size=30, length=470)
BeetleFly = Data_set(name='BeetleFly', classes_num=2, train_size=20, test_size=20, length=512)
BirdChicken = Data_set(name='BirdChicken', classes_num=2, train_size=20, test_size=20, length=512)
BME = Data_set(name='BME', classes_num=3, train_size=30, test_size=150, length=128)
Car= Data_set(name='Car', classes_num=4, train_size=60, test_size=60, length=577)
CBF= Data_set(name='CBF', classes_num=3, train_size=30, test_size=900, length=128)
Chinatown= Data_set(name='Chinatown', classes_num=2, train_size=20, test_size=343, length=24)
ChlorineCon= Data_set(name='ChlorineCon', classes_num=3, train_size=467, test_size=3840, length=166)
CinCECGTorso= Data_set(name='CinCECGTorso', classes_num=4, train_size=40, test_size=1380, length=1639)
Coffee= Data_set(name='Coffee', classes_num=2, train_size=28, test_size=28, length=286)
Computers= Data_set(name='Computers', classes_num=2, train_size=250, test_size=250, length=720)
CricketX= Data_set(name='CricketX', classes_num=12, train_size=390, test_size=390, length=300)
CricketY= Data_set(name='CricketY', classes_num=12, train_size=390, test_size=390, length=300)
CricketZ= Data_set(name='CricketZ', classes_num=12, train_size=390, test_size=390, length=300)
Crop= Data_set(name='Crop', classes_num=24, train_size=7200, test_size=16800, length=46)
DiatomSizeR= Data_set(name='DiatomSizeR', classes_num=4, train_size=16, test_size=306, length=345)
DistalPhalanxOutlineAgeGroup= Data_set(name='DistalPhalanxOutlineAgeGroup', classes_num=3, train_size=400, test_size=139, length=80)
DistalPhalanxOutlineCorrect= Data_set(name='DistalPhalanxOutlineCorrect', classes_num=2, train_size=600, test_size=276, length=80)
DistalPhalanxTW= Data_set(name='DistalPhalanxTW', classes_num=6, train_size=400, test_size=139, length=80)
DodgerLoopDay= Data_set(name='DodgerLoopDay', classes_num=7, train_size=78, test_size=80, length=288)
DodgerLoopGame= Data_set(name='DodgerLoopGame', classes_num=2, train_size=20, test_size=138, length=288)
DodgerLoopWeekend= Data_set(name='DodgerLoopWeekend', classes_num=2, train_size=20, test_size=138, length=288)
Earthquakes= Data_set(name='Earthquakes', classes_num=2, train_size=322, test_size=139, length=512)
ECG200= Data_set(name='ECG200', classes_num=2, train_size=100, test_size=100, length=96)
ECG5000= Data_set(name='ECG5000', classes_num=5, train_size=500, test_size=4500, length=140)
ECGFiveDays= Data_set(name='ECGFiveDays', classes_num=2, train_size=23, test_size=861, length=136)
ElectricDevices= Data_set(name='ElectricDevices', classes_num=7, train_size=8926, test_size=7711, length=96)
EOGHorizontalSignal= Data_set(name='EOGHorizontalSignal', classes_num=12, train_size=362, test_size=362, length=1250)
EOGVerticalSignal= Data_set(name='EOGVerticalSignal', classes_num=12, train_size=362, test_size=362, length=1250)
EthanolLevel= Data_set(name='EthanolLevel', classes_num=4, train_size=504, test_size=500, length=1751)
FaceAll= Data_set(name='FaceAll', classes_num=14, train_size=560, test_size=1690, length=131)
FaceFour= Data_set(name='FaceFour', classes_num=4, train_size=24, test_size=88, length=350)
FacesUCR= Data_set(name='FacesUCR', classes_num=14, train_size=200, test_size=2050, length=131)
Fish= Data_set(name='Fish', classes_num=7, train_size=175, test_size=175, length=463)
FordA= Data_set(name='FordA', classes_num=2, train_size=3601, test_size=1320, length=500)
FordB= Data_set(name='FordB', classes_num=2, train_size=3636, test_size=810, length=500)
FreezerRegularTrain= Data_set(name='FreezerRegularTrain', classes_num=2, train_size=150, test_size=2850, length=301)
FreezerSmallTrain= Data_set(name='FreezerSmallTrain', classes_num=2, train_size=28, test_size=2850, length=301)
Fungi= Data_set(name='Fungi', classes_num=18, train_size=18, test_size=186, length=201)
GestureMidAirD1= Data_set(name='GestureMidAirD1', classes_num=26, train_size=208, test_size=130, length=360)
GestureMidAirD2= Data_set(name='GestureMidAirD2', classes_num=26, train_size=208, test_size=130, length=360)
GestureMidAirD3= Data_set(name='GestureMidAirD3', classes_num=26, train_size=208, test_size=130, length=360)
GesturePebbleZ1= Data_set(name='GesturePebbleZ1', classes_num=6, train_size=132, test_size=172, length=455)
GesturePebbleZ2= Data_set(name='GesturePebbleZ2', classes_num=6, train_size=146, test_size=158, length=455)
GunPoint= Data_set(name='GunPoint', classes_num=2, train_size=50, test_size=150, length=150)
GunPointAgeSpan= Data_set(name='GunPointAgeSpan', classes_num=2, train_size=135, test_size=316, length=150)
GunPointMaleVersusFemale= Data_set(name='GunPointMaleVersusFemale', classes_num=2, train_size=135, test_size=316, length=150)
GunPointOldVersusYoung= Data_set(name='GunPointOldVersusYoung', classes_num=2, train_size=136, test_size=315, length=150)
Ham= Data_set(name='Ham', classes_num=2, train_size=109, test_size=105, length=431)
HandOutlines= Data_set(name='HandOutlines', classes_num=2, train_size=1000, test_size=370, length=2709)
Haptics= Data_set(name='Haptics', classes_num=5, train_size=155, test_size=308, length=1092)
Herring= Data_set(name='Herring', classes_num=2, train_size=64, test_size=64, length=512)
HouseTwenty= Data_set(name='HouseTwenty', classes_num=2, train_size=40, test_size=119, length=2000)
InlineSkate= Data_set(name='InlineSkate', classes_num=7, train_size=100, test_size=550, length=1882)
InsectEPGRegularTrain= Data_set(name='InsectEPGRegularTrain', classes_num=3, train_size=62, test_size=249, length=601)
InsectEPGSmallTrain= Data_set(name='InsectEPGSmallTrain', classes_num=3, train_size=17, test_size=249, length=601)
InsectWingbeatSound= Data_set(name='InsectWingbeatSound', classes_num=11, train_size=220, test_size=1980, length=256)
ItalyPower= Data_set(name='ItalyPower', classes_num=2, train_size=67, test_size=1029, length=24)
LargeKitchenAppliances= Data_set(name='LargeKitchenAppliances', classes_num=3, train_size=375, test_size=375, length=720)
Lighting2= Data_set(name='Lighting2', classes_num=2, train_size=60, test_size=61, length=637)
Lighting7= Data_set(name='Lighting7', classes_num=7, train_size=70, test_size=73, length=319)
MALLAT= Data_set(name='MALLAT', classes_num=8, train_size=55, test_size=2345, length=1024)
Meat= Data_set(name='Meat', classes_num=3, train_size=60, test_size=60, length=448)
MedicalImages= Data_set(name='MedicalImages', classes_num=10, train_size=381, test_size=760, length=99)
MelbournePedestrian= Data_set(name='MelbournePedestrian', classes_num=10, train_size=1194, test_size=2439, length=24)
MiddlePhalanxOutlineAgeGroup= Data_set(name='MiddlePhalanxOutlineAgeGroup', classes_num=3, train_size=400, test_size=154, length=80)
MiddlePhalanxOutlineCorrect= Data_set(name='MiddlePhalanxOutlineCorrect', classes_num=2, train_size=600, test_size=291, length=80)
MiddlePhalanxTW= Data_set(name='MiddlePhalanxTW', classes_num=6, train_size=399, test_size=154, length=80)
MixedShapesRegularTrain= Data_set(name='MixedShapesRegularTrain', classes_num=5, train_size=500, test_size=2425, length=1024)
MixedShapesSmallTrain= Data_set(name='MixedShapesSmallTrain', classes_num=5, train_size=100, test_size=2425, length=1024)
MoteStrain= Data_set(name='MoteStrain', classes_num=2, train_size=20, test_size=1252, length=84)
NonInvThorax1= Data_set(name='NonInvThorax1', classes_num=42, train_size=1800, test_size=1965, length=750)
NonInvThorax2= Data_set(name='NonInvThorax2', classes_num=42, train_size=1800, test_size=1965, length=750)
OliveOil= Data_set(name='OliveOil', classes_num=4, train_size=30, test_size=30, length=570)
OSULeaf= Data_set(name='OSULeaf', classes_num=6, train_size=200, test_size=242, length=427)
PhalangesOutlinesCorrect= Data_set(name='PhalangesOutlinesCorrect', classes_num=2, train_size=1800, test_size=858, length=80)
Phoneme= Data_set(name='Phoneme', classes_num=39, train_size=214, test_size=1896, length=1024)
PickupGestureWiimoteZ= Data_set(name='PickupGestureWiimoteZ', classes_num=10, train_size=50, test_size=50, length=361)
PigAirwayPressure= Data_set(name='PigAirwayPressure', classes_num=52, train_size=104, test_size=208, length=2000)
PigArtPressure= Data_set(name='PigArtPressure', classes_num=52, train_size=104, test_size=208, length=2000)
PigCVP= Data_set(name='PigCVP', classes_num=52, train_size=104, test_size=208, length=2000)
PLAID= Data_set(name='PLAID', classes_num=11, train_size=537, test_size=537, length=1345)
Plane= Data_set(name='Plane', classes_num=7, train_size=105, test_size=105, length=144)
PowerCons= Data_set(name='PowerCons', classes_num=2, train_size=180, test_size=180, length=144)
ProximalPhalanxOutlineAgeGroup= Data_set(name='ProximalPhalanxOutlineAgeGroup', classes_num=3, train_size=400, test_size=205, length=80)
ProximalPhalanxOutlineCorrect= Data_set(name='ProximalPhalanxOutlineCorrect', classes_num=2, train_size=600, test_size=291, length=80)
ProximalPhalanxTW= Data_set(name='ProximalPhalanxTW', classes_num=6, train_size=400, test_size=205, length=80)
RefrigerationDevices= Data_set(name='RefrigerationDevices', classes_num=3, train_size=375, test_size=375, length=720)
Rock= Data_set(name='Rock', classes_num=4, train_size=20, test_size=50, length=2844)
ScreenType= Data_set(name='ScreenType', classes_num=3, train_size=375, test_size=375, length=720)
SemgHandGenderCh2= Data_set(name='SemgHandGenderCh2', classes_num=2, train_size=300, test_size=600, length=1500)
SemgHandMovementCh2= Data_set(name='SemgHandMovementCh2', classes_num=6, train_size=450, test_size=450, length=1500)
SemgHandSubjectCh2= Data_set(name='SemgHandSubjectCh2', classes_num=5, train_size=450, test_size=450, length=1500)
ShakeGestureWiimoteZ= Data_set(name='ShakeGestureWiimoteZ', classes_num=10, train_size=50, test_size=50, length=385)
ShapeletSim= Data_set(name='ShapeletSim', classes_num=2, train_size=20, test_size=180, length=500)
ShapesAll= Data_set(name='ShapesAll', classes_num=60, train_size=600, test_size=600, length=512)
SmallKitchenAppliances= Data_set(name='SmallKitchenAppliances', classes_num=3, train_size=375, test_size=375, length=720)
SmoothSubspace= Data_set(name='SmoothSubspace', classes_num=3, train_size=150, test_size=150, length=15)
SonyAIBORobot= Data_set(name='SonyAIBORobot', classes_num=2, train_size=20, test_size=601, length=70)
SonyAIBORobotII= Data_set(name='SonyAIBORobotII', classes_num=2, train_size=27, test_size=953, length=65)
StarLightCurves= Data_set(name='StarLightCurves', classes_num=3, train_size=1000, test_size=8236, length=1024)
Strawberry= Data_set(name='Strawberry', classes_num=2, train_size=613, test_size=370, length=235)
SwedishLeaf= Data_set(name='SwedishLeaf', classes_num=15, train_size=500, test_size=625, length=128)
Symbols= Data_set(name='Symbols', classes_num=6, train_size=25, test_size=995, length=398)
SyntheticControl= Data_set(name='SyntheticControl', classes_num=6, train_size=300, test_size=300, length=60)
ToeSegmentation1= Data_set(name='ToeSegmentation1', classes_num=2, train_size=40, test_size=228, length=277)
ToeSegmentation2= Data_set(name='ToeSegmentation2', classes_num=2, train_size=36, test_size=130, length=343)
Trace= Data_set(name='Trace', classes_num=4, train_size=100, test_size=100, length=275)
TwoLeadECG= Data_set(name='TwoLeadECG', classes_num=2, train_size=23, test_size=1139, length=82)
TwoPatterns= Data_set(name='TwoPatterns', classes_num=4, train_size=1000, test_size=4000, length=128)
UMD= Data_set(name='UMD', classes_num=3, train_size=36, test_size=144, length=150)
UWaveGestureLibraryAll= Data_set(name='UWaveGestureLibraryAll', classes_num=8, train_size=896, test_size=3582, length=945)
UWaveX= Data_set(name='UWaveX', classes_num=8, train_size=896, test_size=3582, length=315)
UWaveY= Data_set(name='UWaveY', classes_num=8, train_size=896, test_size=3582, length=315)
UWaveZ= Data_set(name='UWaveZ', classes_num=8, train_size=896, test_size=3582, length=315)
Wafer= Data_set(name='Wafer', classes_num=2, train_size=1000, test_size=6147, length=152)
Wine= Data_set(name='Wine', classes_num=2, train_size=57, test_size=54, length=234)
Words50 = Data_set(name='Words50', classes_num=50, train_size=450, test_size=455,length=270)
WordSynonyms = Data_set(name='WordSynonyms', classes_num=25, train_size=267, test_size=638,length=270)
Worms = Data_set(name='Worms', classes_num=5, train_size=181, test_size=77,length=900)
WormsTwoClass = Data_set(name='WormsTwoClass', classes_num=2, train_size=181, test_size=77,length=900)
Yoga= Data_set(name='Yoga', classes_num=2, train_size=300, test_size=3000, length=426)

data_set_dict = {'ACSF1':ACSF1,'Adiac':Adiac,'AllGestureWiimoteX':AllGestureWiimoteX,'AllGestureWiimoteY':AllGestureWiimoteY,
                 'AllGestureWiimoteZ':AllGestureWiimoteZ,'ArrowHead':ArrowHead,'Beef': Beef,'BeetleFly': BeetleFly,
                 'BirdChicken':BirdChicken,'BME':BME,'Car':Car,'CBF': CBF,'Chinatown': Chinatown,'ChlorineCon': ChlorineCon,
                 'CinCECGTorso': CinCECGTorso,'Coffee': Coffee,'Computers':Computers,'CricketX': CricketX,'CricketY': CricketY,
                 'CricketZ': CricketZ,'Crop': Crop,'DiatomSizeR': DiatomSizeR,'DistalPhalanxOutlineAgeGroup':DistalPhalanxOutlineAgeGroup,
                 'DistalPhalanxOutlineCorrect':DistalPhalanxOutlineCorrect,'DistalPhalanxTW':DistalPhalanxTW,'DodgerLoopDay':DodgerLoopDay,
                 'DodgerLoopGame':DodgerLoopGame,'DodgerLoopWeekend':DodgerLoopWeekend,'Earthquakes':Earthquakes,
                 'ECG200':ECG200,'ECG5000':ECG5000,'ECGFiveDays': ECGFiveDays,'ElectricDevices':ElectricDevices,'EOGHorizontalSignal':EOGHorizontalSignal,
                 'EOGVerticalSignal':EOGVerticalSignal,'EthanolLevel':EthanolLevel,'FaceAll': FaceAll,'FaceFour': FaceFour,'FacesUCR': FacesUCR,'Fish': Fish,
                 'FordA':FordA,'FordB':FordB,'FreezerRegularTrain':FreezerRegularTrain,'FreezerSmallTrain':FreezerSmallTrain,'Fungi':Fungi,
                 'GestureMidAirD1':GestureMidAirD1,'GestureMidAirD2':GestureMidAirD2,'GestureMidAirD3':GestureMidAirD3,'GesturePebbleZ1':GesturePebbleZ1,
                 'GesturePebbleZ2':GesturePebbleZ2,'GunPoint': GunPoint,'GunPointAgeSpan': GunPointAgeSpan,'GunPointMaleVersusFemale': GunPointMaleVersusFemale,
                 'GunPointOldVersusYoung': GunPointOldVersusYoung,'Ham':Ham,'HandOutlines':HandOutlines,'Haptics': Haptics,'Herring':Herring,
                 'HouseTwenty':HouseTwenty,'InlineSkate':InlineSkate,'InsectEPGRegularTrain':InsectEPGRegularTrain,'InsectEPGSmallTrain':InsectEPGSmallTrain,
                 'InsectWingbeatSound':InsectWingbeatSound,'ItalyPower':ItalyPower,'LargeKitchenAppliances':LargeKitchenAppliances,'Lighting2':Lighting2,
                 'Lighting7':Lighting7,'MALLAT':MALLAT,'Meat':Meat,'MedicalImages': MedicalImages,'MelbournePedestrian': MelbournePedestrian,
                 'MiddlePhalanxOutlineAgeGroup':MiddlePhalanxOutlineAgeGroup,'MiddlePhalanxOutlineCorrect':MiddlePhalanxOutlineCorrect,
                 'MiddlePhalanxTW':MiddlePhalanxTW,'MixedShapesRegularTrain':MixedShapesRegularTrain,'MixedShapesSmallTrain':MixedShapesSmallTrain,
                 'MoteStrain': MoteStrain,'NonInvThorax1': NonInvThorax1,'NonInvThorax2': NonInvThorax2,'OliveOil': OliveOil,'OSULeaf': OSULeaf,
                 'PhalangesOutlinesCorrect':PhalangesOutlinesCorrect,'Phoneme':Phoneme,'PickupGestureWiimoteZ':PickupGestureWiimoteZ,
                 'PigAirwayPressure':PigAirwayPressure,'PigArtPressure':PigArtPressure,'PigCVP':PigCVP,'PLAID':PLAID,'Plane':Plane,'PowerCons':PowerCons,
                 'ProximalPhalanxOutlineAgeGroup':ProximalPhalanxOutlineAgeGroup,'ProximalPhalanxOutlineCorrect':ProximalPhalanxOutlineCorrect,
                 'ProximalPhalanxTW':ProximalPhalanxTW,'RefrigerationDevices':RefrigerationDevices,'Rock':Rock,'ScreenType':ScreenType,
                 'SemgHandGenderCh2':SemgHandGenderCh2,'SemgHandMovementCh2':SemgHandMovementCh2,'SemgHandSubjectCh2':SemgHandSubjectCh2,
                 'ShakeGestureWiimoteZ':ShakeGestureWiimoteZ,'ShapeletSim':ShapeletSim,'ShapesAll':ShapesAll,'SmallKitchenAppliances':SmallKitchenAppliances,
                 'SmoothSubspace':SmoothSubspace,'SonyAIBORobot': SonyAIBORobot,'SonyAIBORobotII': SonyAIBORobotII,'StarLightCurves': StarLightCurves,'Strawberry':Strawberry,
                 'SwedishLeaf': SwedishLeaf,'Symbols': Symbols,'SyntheticControl': SyntheticControl,'ToeSegmentation1':ToeSegmentation1,
                 'ToeSegmentation2':ToeSegmentation2,'Trace': Trace,'TwoLeadECG': TwoLeadECG,'TwoPatterns': TwoPatterns,'UMD': UMD,
                 'UWaveGestureLibraryAll': UWaveGestureLibraryAll,'UWaveX': UWaveX,'UWaveY': UWaveY,'UWaveZ': UWaveZ,'Wafer': Wafer,'Wine':Wine,'Words50':Words50,
                 'WordSynonyms':WordSynonyms,'Worms':Worms,'WormsTwoClass':WormsTwoClass,'Yoga': Yoga}








