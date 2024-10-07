# How to download our datasets

This document provides an explanation of how to use get_datasets.py to download the three datasets provided in this project: the South Kensington Stereo Vision dataset, Infinigen Stereo and InfinigenSV.

## General structure

```
python3 get_data_all.py --download_path /local/download/path --datasets SouthKensington InfinigenStereo InfinigenSV --SK_all --InfinigenStereo_all --InfinigenSV_all
```

* --download_path [required]. It specifies where on your local machine the datasets directory will be downloaded to.
* --datasets [optional]. It allows you to specify which of the three datasets you would like to download. You can download just one of them, two out of the three or all of them.

Note: For the following options, SouthKensigton must first be specified in --datasets.
* --SK_images [optional]. A flag that specifies that the images in the South Kensington dataset should be downloaded.
* --SK_videos [optional]. A flag that specifies that the videos in the South Kensington dataset should be downloaded.
* --SK_tracking_data [optional]. A flag that specifies that the tracking data files in the South Kensington dataset should be downloaded (Tracking data is only provided in the indoor videos).
* --SK_specify_vid_ids [optional]. A flag that specifies that the user wants to select the videos that out downloaded manually, via video id.
* --SK_indoor_ids [optional]. A list of indoor ids to be downloaded.
* --SK_outdoor_ids [optional]. A list of outdoor ids to be downloaded.
* --SK_all [optional]. A flag that specifies that everything in the South Kensington dataset should be downloaded. Makes the previous SK flags irrelevant.

## Downloading everything

```
python3 get_data_all.py --download_path /local/download/path --datasets SouthKensington InfinigenStereo InfinigenSV --SK_all --InfinigenStereo_all --InfinigenSV_all
```
