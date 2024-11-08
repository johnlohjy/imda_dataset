# File Directory

**Focus on**

```prepare_dataset.ipynb```: Prepare the dataset to be uploaded to HF

```imda_nsc_testing.ipynb```: Dataset loading script to be uploaded to HF

```use_dataset.ipynb```: Short example on how to stream from the dataset

<br/>
<br/>
<br/>

# Steps

### Step 1: Create a new HF dataset repository

![alt text](images/HF_create_dataset_repo.png)

### Step 2: Follow the instructions and run ```prepare_dataset.ipynb```
- Just take note of the parts with **USER INPUT REQUIRED** in Step 1 only
    - Change the relative paths and naming conventions if you want (but not needed)
    - Add in the ```.wav``` and ```.TextGrid``` files from IMDA NSC into ```org_waves``` and ```org_transcript``` respectively after the directory has been initialised
        - Use <u>Audio Same CloseMic</u> folder for the ```.wav``` files first (from Part 3 in the DropBox)
        - Use <u>Scripts Same</u> folder for the ```.TextGrid``` files first (from Part 3 in the DropBox)

### Step 3: Upload the ```data``` folder containing the compressed files to the created dataset repository

![alt text](images/local_folder_structure_1.png)

![alt text](images/local_folder_structure_2.png)

![alt text](images/HF_upload_compressed_files.png)

![alt text](images/HF_upload_compressed_files_3.png)

![alt text](images/HF_upload_compressed_files_2.png)

### Step 4: Modify and upload the loading script as needed. Note that it has to have the same name as the created HF dataset repo

Example if your HF dataset repo is called ```imda_nsc_testing```, your loading script has to be called ```imda_nsc_prototype.py```

Modify the paths in the loading script as needed

![alt text](images/HF_upload_loading_script.png)

### Step 5: Refer to ```use_dataset.ipynb``` on how to stream the dataset and its splits