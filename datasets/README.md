# How to download the datasets

Complete the [form](https://docs.google.com/forms/d/e/1FAIpQLScBAqAtfXKtp9lxg1neGv0YyPV4EYNKP9UHQ5TRORirDMtwXw/viewform) to get the Python script for downloading the following datasets: South Kensington Stereo Vision dataset, Infinigen Stereo and InfinigenSV.
  

```
python3 get_datasets.py --download_path /local/download/path --datasets SouthKensington InfinigenStereo InfinigenSV --SK_all --InfinigenStereo_all --InfinigenSV_all
```

The above code example downloads everything from all three datasets.

Below is a description of all available flags:

* `--download_path` [required]. It specifies where on your local machine the datasets directory will be downloaded to.
* `--datasets` [optional]. It allows you to specify which of the three datasets you would like to download. You can download just one of them, two out of the three or all of them.

Note: For the following options, SouthKensigton must first be specified in --datasets.
* `--SK_images` [optional]. A flag that specifies that the images in the South Kensington dataset should be downloaded.
* `--SK_videos` [optional]. A flag that specifies that the videos in the South Kensington dataset should be downloaded.
* `--SK_tracking_data` [optional]. A flag that specifies that the tracking data files in the South Kensington dataset should be downloaded (Tracking data is only provided in the indoor videos).
* `--SK_specify_vid_ids` [optional]. A flag that specifies that the user wants to select the sequences to downloaded manually, via id.
* `--SK_indoor_ids` [optional]. A list of indoor ids to be downloaded.
* `--SK_outdoor_ids` [optional]. A list of outdoor ids to be downloaded.
* `--SK_all` [optional]. A flag that specifies that everything in the South Kensington dataset should be downloaded. Makes the previous SK flags irrelevant.

Note: For the following options, InfinigenStereo must first be specified in --datasets.
* `--InfinigenStereo_images` [optional]. A flag that specifies that the images in the Infinigen Stereo dataset should be downloaded.
* `--InfinigenStereo_depths` [optional]. A flag that specifies that the depth images in the Infinigen Stereo dataset should be downloaded.
* `--InfinigenStereo_camviews` [optional]. A flag that specifies that the camviews in the Infinigen Stereo dataset should be downloaded.
* `--InfinigenStereo_videos` [optional]. A flag that specifies that the videos in the Infinigen Stereo dataset should be downloaded.
* `--InfinigenStereo_specify_vid_ids` [optional]. A flag that specifies that the user wants to select the sequences to downloaded manually, via id.
* `--InfinigenStereo_vid_ids` [optional]. A list of ids to be downloaded.
* `--InfinigenStereo_all` [optional]. A flag that specifies that everything in the Infinigen Stereo dataset should be downloaded. Makes the previous InfinigenStereo flags irrelevant.

Note: For the following options, InfinigenSV must first be specified in --datasets.
* `--InfinigenSV_videos` [optional]. A flag that specifies that the videos in the Infinigen Stereo dataset should be downloaded.
* `--InfinigenSV_specify_vid_ids` [optional]. A flag that specifies that the user wants to select the sequences to downloaded manually, via id.
* `--InfinigenSV_vid_ids` [optional]. A list of ids to be downloaded.
* `--InfinigenSV_all` [optional]. A flag that specifies that everything in the InfinigenSV dataset should be downloaded. Makes the previous InfinigenSV flags irrelevant.

SouthKensington SV dataset is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). Under this license, if you want to modify SK-SV or generate new data from SK-SV dataset (e.g., super-resolution, denoising, defocus), the releasing of your new data should be licensed under the same CC BY-NC-SA 4.0. 

## Citation 
If you use these datasets in your research, please use the following BibTeX entry.
```
@article{jing2024match,
  title={Match Stereo Videos via Bidirectional Alignment},
  author={Junpeng Jing and Ye Mao and Anlan Qiu and Krystian Mikolajczyk},
  year={2024}
}

```
