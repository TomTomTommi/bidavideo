import argparse
import os
import subprocess

def partition_list(big_list, num1, num2):
    # Initialize the three partitions as empty lists
    list1 = []
    list2 = []
    list3 = []

    # Iterate over the big list and categorize each element
    for item in big_list:
        if item <= num1:
            list1.append(item)
        elif num1 < item <= num2:
            list2.append(item)
        else:
            list3.append(item)

    return list1, list2, list3
    
def download_sk_indoor(output_directory, images_flag, videos_flag, tracking_flag, specify_vids_flag, ids):
    output_directory = os.path.join(output_directory, 'Indoor')
    os.makedirs(output_directory, exist_ok=True)

    if specify_vids_flag is True:
        video_ids = ids
    else:
        video_ids = list(range(1, 36))

    for i in video_ids:
        folder_name = f'video{i:03d}/'

        scene_dir = os.path.join(output_directory, folder_name)
        try:
            os.mkdir(scene_dir)
            print(f"Directory '{scene_dir}' created")
        except FileExistsError:
            print(f"Directory '{scene_dir}' already exists")

        if images_flag:
            command = ['wget', '-P', scene_dir, 'matchlab-web.dept.ic.ac.uk/junpeng/SouthKensington/Indoor/'+folder_name+'images.zip']
            result = subprocess.run(command, capture_output=True, text=True)

            zip_path = os.path.join(scene_dir, 'images.zip')
            command2 = ['unzip', zip_path, '-d', scene_dir]
            result2 = subprocess.run(command2, capture_output=True, text=True)
            os.remove(zip_path)


            if result.returncode == 0:
                print("Image directory successfully downloaded")
            else:
                print("Image directory download failed")

        if videos_flag:
            command = ['wget', '-P', scene_dir, 'matchlab-web.dept.ic.ac.uk/junpeng/SouthKensington/Indoor/'+folder_name+'video.mp4']
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                print("Error occurred when downloading video: ", result.stderr)
            else:
                print("Video downloaded")

        if tracking_flag:
            command = ['wget', '-P', scene_dir, 'matchlab-web.dept.ic.ac.uk/junpeng/SouthKensington/Indoor/'+folder_name+'tracking_data.txt']
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                print("Error occurred when downloading tracking data: ", result.stderr)
            else:
                print("Tracking data downloaded")

def download_sk_outdoor(output_directory, images_flag, videos_flag, specify_vids_flag, ids):

    output_directory = os.path.join(output_directory, 'Outdoor')
    os.makedirs(output_directory, exist_ok=True)

    if specify_vids_flag is True:
        video_ids = ids
    else:
        video_ids = list(range(1, 230))

    for i in video_ids:
        folder_name = f'video{i:03d}/'

        scene_dir = os.path.join(output_directory, folder_name)
        try:
            os.mkdir(scene_dir)
            print(f"Directory '{scene_dir}' created")
        except FileExistsError:
            print(f"Directory '{scene_dir}' already exists")

        if images_flag:
            command = ['wget', '-P', scene_dir, 'matchlab-web.dept.ic.ac.uk/junpeng/SouthKensington/Outdoor/'+folder_name+'images.zip']
            result = subprocess.run(command, capture_output=True, text=True)

            zip_path = os.path.join(scene_dir, 'images.zip')
            command2 = ['unzip', zip_path, '-d', scene_dir]
            result2 = subprocess.run(command2, capture_output=True, text=True)
            os.remove(zip_path)


            if result.returncode == 0:
                print("Image directory successfully downloaded")
            else:
                print("Image directory download failed")

        if videos_flag:
            command = ['wget', '-P', scene_dir, 'matchlab-web.dept.ic.ac.uk/junpeng/SouthKensington/Outdoor/'+folder_name+'video.mp4']
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                print("Error occurred when downloading video: ", result.stderr)
            else:
                print("Video downloaded")


def download_sk(output_directory, images_flag, videos_flag, tracking_flag, specify_vids_flag, indoor_ids, outdoor_ids):
    output_directory = os.path.join(output_directory, 'sk_dataset')
    os.makedirs(output_directory, exist_ok=True)
    download_sk_indoor(output_directory, images_flag, videos_flag, tracking_flag, specify_vids_flag, indoor_ids)
    download_sk_outdoor(output_directory, images_flag, videos_flag, specify_vids_flag, outdoor_ids)


def download_InfinigenStereo(output_directory, image_flag, depth_flag, camview_flag, videos_flag, specify_vids_flag, ids):

    output_directory = os.path.join(output_directory, 'InfinigenStereo')
    os.makedirs(output_directory, exist_ok=True)

    test_directory = os.path.join(output_directory, 'test')
    os.makedirs(test_directory, exist_ok=True)
    train_directory = os.path.join(output_directory, 'train')
    os.makedirs(train_directory, exist_ok=True)
    val_directory = os.path.join(output_directory, 'val')
    os.makedirs(val_directory, exist_ok=True)


    if specify_vids_flag is True:
        test_ids, train_ids, val_ids = partition_list(ids, 30, 216)
    else:
        test_ids = list(range(1, 31))
        train_ids = list(range(31, 217))
        val_ids = list(range(217, 227))

    test_train_val = [
        (test_ids, test_directory, 'test'),
        (train_ids, train_directory, 'train'),
        (val_ids, val_directory, 'val')
    ]

    for id_type, d_type, ttv in test_train_val:
        for i in id_type:
            folder_name = f'scene{i:03d}/'

            scene_dir = os.path.join(d_type, folder_name)
            try:
                os.mkdir(scene_dir)
                print(f"Directory '{scene_dir}' created")
            except FileExistsError:
                print(f"Directory '{scene_dir}' already exists")

            folder_name = f'scene{i:03d}/frames/'
            scene_dir = os.path.join(scene_dir, 'frames/')
            os.mkdir(scene_dir)

            file_data = [
                (image_flag, 'Image.zip', 'Image'),
                (depth_flag, 'Depth.zip', 'Depth'),
                (camview_flag, 'camview.zip', 'Camview')
            ]

            for flag, zip_file, dir_name in file_data:
                if flag:
                    command = ['wget', '-P', scene_dir, 'matchlab-web.dept.ic.ac.uk/junpeng/InfinigenStereo/'+ttv+'/'+folder_name+zip_file]
                    result = subprocess.run(command, capture_output=True, text=True)

                    zip_path = os.path.join(scene_dir, zip_file)
                    command2 = ['unzip', zip_path, '-d', scene_dir]
                    result2 = subprocess.run(command2, capture_output=True, text=True)
                    os.remove(zip_path)

                    if result.returncode == 0:
                        print(f"'{dir_name}' directory download successful")
                    else:
                        print(f"'{dir_name}' directory download failed")
            
            if videos_flag:
                folder_name = f'scene{i:03d}'
                command = ['wget', '-P', scene_dir, 'matchlab-web.dept.ic.ac.uk/junpeng_2/mp4s_infinigen/('+folder_name+')_depth.mp4']
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode != 0:
                    print("Error occurred when downloading depth video: ", result.stderr)
                else:
                    print("Depth video downloaded")
                
                command = ['wget', '-P', scene_dir, 'matchlab-web.dept.ic.ac.uk/junpeng_2/mp4s_infinigen/('+folder_name+')_image.mp4']
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode != 0:
                    print("Error occurred when downloading image video: ", result.stderr)
                else:
                    print("Image video downloaded")

def download_InfinigenSV(output_directory, videos_flag, specify_vids_flag, ids):
    output_directory = os.path.join(output_directory, 'InfinigenSV')
    os.makedirs(output_directory, exist_ok=True)

    if specify_vids_flag is True:
        video_ids = ids
    else:
        video_ids = [8]

    for i in video_ids:
        folder_name = f'scene{i:03d}/'

        scene_dir = os.path.join(output_directory, folder_name)
        try:
            os.mkdir(scene_dir)
            print(f"Directory '{scene_dir}' created")
        except FileExistsError:
            print(f"Directory '{scene_dir}' already exists")
        
        folder_name = f'scene{i:03d}/frames/'
        scene_dir = os.path.join(output_directory, folder_name)
        os.mkdir(scene_dir)

        file_data = ['AO.zip', 'camview.zip', 'Depth.zip', 'DiffCol.zip', 'DiffDir.zip', 'DiffInd.zip', 'Emit.zip', 'Env.zip', 'Flow.zip', 'GlossCol.zip', 'GlossDir.zip', 'GlossInd.zip', 'Image.zip', 'InstanceSegmentation.zip', 'Objects.zip', 'ObjectSegmentation.zip', 'SurfaceNormal.zip', 'TransCol.zip', 'TransDir.zip', 'TransInd.zip', 'VolumeDir.zip']

        for zip_file in file_data:
            print(zip_file)
            command = ['wget', '-P', scene_dir, 'matchlab-web.dept.ic.ac.uk/junpeng_2/InfinigenSV/'+folder_name+zip_file]
            result = subprocess.run(command, capture_output=True, text=True)

            zip_path = os.path.join(scene_dir, zip_file)
            command2 = ['unzip', zip_path, '-d', scene_dir]
            result2 = subprocess.run(command2, capture_output=True, text=True)
            os.remove(zip_path)

            dir_name = zip_file[:-4]
            if result.returncode == 0:
                print(f"'{dir_name}' directory download successful")
            else:
                print(f"'{dir_name}' directory download failed")

        if videos_flag:
            print('VIDEO')
            command = ['wget', '-P', scene_dir, 'matchlab-web.dept.ic.ac.uk/junpeng/SouthKensington/Outdoor/'+folder_name+'video.mp4']
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                print("Error occurred when downloading video: ", result.stderr)
            else:
                print("Video downloaded")    


# Step 1: Create the main parser
parser = argparse.ArgumentParser(description="A script to download multiple datasets")

parser.add_argument("--download_path", type=str, required=True, help="Location where datasets will be downloaded to")
parser.add_argument("--datasets", nargs="+", choices=["SouthKensington", "InfinigenStereo", "InfinigenSV"],
                    help="Specify one or more datasets to download")

# Arguments for dataset1

parser.add_argument("--SK_images", action="store_true", help="Save SK images")
parser.add_argument("--SK_videos", action="store_true", help="Save SK videos")
parser.add_argument("--SK_tracking_data", action="store_true", help="Save SK tracking data")

parser.add_argument("--SK_specify_vid_ids", action="store_true", help="Only download specific sequences")
parser.add_argument("--SK_indoor_ids", nargs="+", type=int, help="A list of the ids of indoor videos to download")
parser.add_argument("--SK_outdoor_ids", nargs="+", type=int, help="A list of the ids of outdoor videos to download")
parser.add_argument("--SK_all", action="store_true", help="Download the entire South Kensington dataset")


# Arguments for dataset2

parser.add_argument("--InfinigenStereo_images", action="store_true", help="Save InfinigenStereo images")
parser.add_argument("--InfinigenStereo_depths", action="store_true", help="Save InfinigenStereo depths")
parser.add_argument("--InfinigenStereo_camviews", action="store_true", help="Save InfinigenStereo camviews")
parser.add_argument("--InfinigenStereo_videos", action="store_true", help="Save InfinigenStereo videos")
parser.add_argument("--InfinigenStereo_all", action="store_true", help="Download the entire InfinigenStereo dataset")


parser.add_argument("--InfinigenStereo_specify_vid_ids", action="store_true", help="Only download specific sequences")
parser.add_argument("--InfinigenStereo_vid_ids", nargs="+", type=int, help="A list of ids")

# Arguments for dataset3

parser.add_argument("--InfinigenSV_videos", action="store_true", help="Save InfinigenSV videos")

parser.add_argument("--InfinigenSV_specify_vid_ids", action="store_true", help="Only download specific sequences")
parser.add_argument("--InfinigenSV_vid_ids", nargs="+", type=int, help="A list of ids")
parser.add_argument("--InfinigenSV_all", action="store_true", help="Download the entire InfinigenSV dataset")

# Parse the arguments
args = parser.parse_args()

# Validate datasets and arguments
if not args.datasets:
    parser.error("At least one dataset must be specified using --datasets.")


output_directory = args.download_path
output_directory = os.path.join(output_directory, 'BIDADatasets')
os.makedirs(output_directory, exist_ok=True)


# Step 6: Process the datasets
for dataset in args.datasets:
    if dataset == "SouthKensington":
        if args.SK_all == True:
            download_sk(output_directory, True, True, True, False, [], [])
        else:
            download_sk(output_directory, args.SK_images, args.SK_videos, args.SK_tracking_data, args.SK_specify_vid_ids, args.SK_indoor_ids, args.SK_outdoor_ids)

    elif dataset == "InfinigenStereo":
        if args.InfinigenStereo_all == True:
            download_InfinigenStereo(output_directory, True, True, True, True, False, [])
        else:
            download_InfinigenStereo(output_directory, args.InfinigenStereo_images, args.InfinigenStereo_depths, args.InfinigenStereo_camviews, args.InfinigenStereo_videos, args.InfinigenStereo_speify_vid_ids, args.InfinigenStereo_vid_ids)

    elif dataset == "InfinigenSV":
        if args.InfinigenSV_all == True:
            download_InfinigenSV(output_directory, True, False, [])
        else:
            download_InfinigenSV(output_directory, args.InfinigenSV_vid_ids, args.InfinigenSV_speify_vid_ids, args.InfinigenSV_vid_ids)
