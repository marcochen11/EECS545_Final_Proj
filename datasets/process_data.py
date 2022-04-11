from pathlib import Path

import argparse
import h5py
import nibabel
import numpy
import os


# TODO: how to download testdata?
def main(args: argparse.Namespace):
    # Assuming filename are sth. like 'DET0000101_avg.nii' or 'DET0000101_avg_seg.nii'
    filenames = os.listdir(args.original_dataset_dir)
    # print(filenames)
    filename_stems = set([file.stem.split('_')[0] for file in args.original_dataset_dir.iterdir()])
    counter = 0
    for filename_stem in filename_stems:
        case_id = filename_stem[-4:]

        image_path = Path(args.original_dataset_dir / f'{filename_stem}_avg.nii.gz')
        label_path = Path(args.original_dataset_dir / f'{filename_stem}_avg_seg.nii.gz')
        # assert image_path.exists() and label_path.exists(), f'For id {filename_stem} either the image or label file ' \
        #                                                     f'is missing'
        if not (image_path.exists() and label_path.exists()):
            print(filename_stem)
            continue
        image_data = nibabel.load(image_path).get_fdata()
        label_data = nibabel.load(label_path).get_fdata()

        normalised_image_data = image_data / 255

        # Reorders data so that the channel dimension is at the front for easier indexing later
        transposed_image_data = numpy.transpose(normalised_image_data, (2, 0, 1))
        transposed_label_data = numpy.transpose(label_data, (2, 0, 1))

        # Extracting slices for training
        for i, (image_slice, label_slice) in enumerate(zip(transposed_image_data, transposed_label_data)):
            if case_id == "0001":
                print(filename_stem)
            out_filename = args.target_dataset_dir / f'Synapse/train_npz/case{case_id}_slice{i:03d}.npz'
            if not out_filename.parent.exists():
                out_filename.parent.mkdir(exist_ok=True, parents=True)
            numpy.savez(out_filename, image=image_slice, label=label_slice)


        # keep the 3D volume in h5 format for testing cases.
        # TODO: check if this is correct or if the testdata should be downloaded separately
        h5_filename = args.target_dataset_dir / f'Synapse/test_vol_h5/case{case_id}.npy.h5'
        if not h5_filename.parent.exists():
            h5_filename.parent.mkdir(exist_ok=True, parents=True)
        with h5py.File(h5_filename, 'w') as f:
            f.create_dataset('image', data=normalised_image_data)
            f.create_dataset('label', data=label_data)
    print(counter)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('original_dataset_dir', type=Path,
                        help='The root directory for the downloaded, original dataset')
    parser.add_argument('-td', '--target_dataset_dir', type=Path, default=Path('../../data'),
                        help='The directory where the processed dataset should be stored.')
    parsed_args = parser.parse_args()
    main(parsed_args)