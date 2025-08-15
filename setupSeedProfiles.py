import os, sys, shutil, copy, re, json
from services import FileOps, Universal, Logger
from addons import ArchSmith, ASReport, ASTracer
from firebase import FireConn
from fm import File, FileManager
from schemas import DI, Figure, Face
from HFProcessor import HFProcessor

FireConn.connect()
DI.setup()
print("FM", FileManager.setup())
print("HFP", HFProcessor.setup())

files = os.listdir('seedFaces')
# try:
#     files.remove('data.json')
# except:
#     print("Data.json not found, didn't remove")

# # Remove numbers + .jpg, keeping only the name part
# clean_names = [re.sub(r'\s+\d+\.jpg$', '.jpg', f) for f in files]  # remove trailing numbers before extension
# clean_names = [name.replace('.jpg', '') for name in clean_names]  # remove extension

# # Get unique names while preserving order
# unique_names = list(dict.fromkeys(clean_names))

# print(unique_names)

# seedFaceNames = ['Tay Beng Chuan', 'Ko Teck Kin', 'Lee Kong Chian', 'Lee Choon Seng', 'Wee Cho Yaw', 'Goh Siew Tin', 'Tan Eng Joo', 'Linn In Hua', 'Teo Siong Seng', 'Lim Kee Ming', 'Soon Peng Yam', 'Ng San Tiong', 'Tan Siak Kew', 'Lee Wee Nam', 'Lien Ying Chow', 'Thomas Chua Kee Seng', 'Kwek Leng Joo', 'Tan Keong Choon', 'Tan Siak Kiew', 'Lim Kee Meng', 'Chua Thian Poh']
# seedFaceData = {}

# count = 0
# for name in seedFaceNames:
#     matches = [x for x in files if name in x]
#     if files:
#         seedFaceData[name] = matches
#         count += len(matches)

# print(seedFaceData, count)

# json.dump(seedFaceData, open('seedFaces/data.json', 'w'), indent=4)

seedFaceData: dict | None = None
with open('seedFaces/data.json', 'r') as f:
    seedFaceData = json.load(f)

if not isinstance(seedFaceData, dict):
    raise Exception("Seed face data load failed. Result:", seedFaceData)

for name in list(seedFaceData.keys()):
    if not isinstance(seedFaceData[name], list):
        print("Removing face '{}' due to invalid value.".format(name))
        del seedFaceData[name]
    
    for file in list(seedFaceData[name]):
        if not isinstance(file, str) or not FileOps.exists(os.path.join('seedFaces', file), type="file"):
            print("Removing invalid file '{}' associated with name '{}'".format(file, name))
            seedFaceData[name].remove(file)
    
    if len(seedFaceData[name]) == 0:
        print("Removing face '{}' due to no valid files.".format(name))
        del seedFaceData[name]

t = ArchSmith.newTracer("Seed Profiles Setup")
for name in list(seedFaceData.keys()):
    t.addReport(
        ASReport(
            "SETUPSEEDPROFILES",
            "Running first image embedding discovery for seed face '{}'.".format(name),
            extraData={"matchedFiles": seedFaceData[name]}
        )
    )
    
    success = []
    firstImageResults = HFProcessor.identifyFaces(os.path.join('seedFaces', seedFaceData[name][0]), t)
    if len(firstImageResults) == 0:
        t.addReport(
            ASReport(
                "SETUPSEEDPROFILES WARNING",
                "No faces detected in the first image for seed face '{}'. Entire face will be skipped.".format(name)
            )
        )
        print("No faces detected in the first image for seed face '{}'. Entire face will be skipped.".format(name))
        continue
    else:
        success.append(seedFaceData[name][0])
    
    targetFigureID = firstImageResults[0]
    t.addReport(
        ASReport(
            "SETUPSEEDPROFILES",
            "Identified figure ID '{}' for seed face '{}'.".format(targetFigureID, name),
            extraData={"matchedFiles": seedFaceData[name]}
        )
    )
    
    figure = Figure.load(targetFigureID)
    figure.label = name
    figure.save(embeds=False)
    del figure
    
    for file in seedFaceData[name][1::]:
        outputIDs = HFProcessor.identifyFaces(os.path.join('seedFaces', file), t, targetFigureID)
        if len(outputIDs) == 0 or outputIDs[0] != targetFigureID:
            t.addReport(
                ASReport(
                    "SETUPSEEDPROFILES WARNING",
                    "Unexpected figure ID '{}' returned in direct face embedding association for file '{}' of name '{}'. Continuing anyways.".format(outputIDs[0] if len(outputIDs) > 0 else "None", file, name),
                    extraData={"matchedFiles": seedFaceData[name]}
                )
            )
            continue
        else:
            success.append(file)
            print("Processed file '{}' successfully.".format(file))
    
    t.addReport(
        ASReport(
            "SETUPSEEDPROFILES",
            "Successfully processed face '{}'. Figure ID: '{}'".format(name, targetFigureID),
            extraData={"success": success, "matchedFaces": seedFaceData[name]}
        )
    )
    print("Face '{}' processed. Figure: '{}'".format(name, targetFigureID))

t.end()
ArchSmith.persist()
print("Setup seed profiles procedure complete.")