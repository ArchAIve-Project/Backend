# Setting up ArchAIve Backend

Guide for future use.

Steps:

1. Clone the codebase into your desired directory.

```zsh
git clone https://github.com/ArchAIve-Project/Backend
```

2. Setup dependencies. Recommended to setup a Python virtual environment prior to this.

```zsh
pip install -r requirements.txt
```

3. Setup `ModelStore` context file at `./models/context.json`. Declare model files that need to exist, including `driveID` for Google Drive download. System requires the following model files minimally. You need to obtain drive ID yourself.

```json
{
    "imageCaptionerResNet": {
        "name": "imageCaptionerResNet",
        "filename": "SCCCICaptionerResNet.pth",
        "driveID": null
    },
    "imageCaptionerViT": {
        "name": "imageCaptionerViT",
        "filename": "SCCCICaptionerViT.pth",
        "driveID": null
    },
    "cnn": {
        "name": "cnn",
        "filename": "CNNBinary.pth",
        "driveID": null
    },
    "ccr": {
        "name": "ccr",
        "filename": "CCR.pth",
        "driveID": null
    },
    "ccrCharFilter": {
        "name": "ccrCharFilter",
        "filename": "CCRCharFilter.pth",
        "driveID": null
    },
    "captionerVocab": {
        "name": "captionerVocab",
        "filename": "vocab.pkl",
        "driveID": null
    }
}
```

4. Optional: Obtain `seedFaces.zip`, extract and place folder at `./seedFaces`. Ensure that `./seedFaces/data.json` exists. Then, run the algorithm to set up seed profiles.

```zsh
python setupSeedProfiles.py
```

5. Set up Firebase project in [console](https://console.firebase.google.com). Obtain service account key and place at `./serviceAccountKey.json`. Create RTDB instance and storage bucket and copy down RTDB URL and storage bucket URL for later environment configuration.

6. Create [Alibaba Cloud](https://alibabacloud.com) and [OpenAI platform](https://platform.openai.com) accounts and obtain Alibaba Model Studio and OpenAI API keys respectively. Copy them down for later environment configuration.

7. Optional: Create an [app password for a Google Account](https://myaccount.google.com/apppasswords) to be used for emailing. Do note that 2FA needs to be setup before an app password can be created. Note down email and app password of account for later environment configuration.

6. Duplicate `.env.example` and rename the new file as `.env`. Fill out placeholders and adjust other settings as needed.

7. Run server.

```zsh
python main.py
```

8. Optional: Run ArchSmith Visualiser.

```zsh
python ASVis.py
```

© 2025 The ArchAIve Team. All rights reserved.