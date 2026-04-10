import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def main():
    mask_path = 'Combined_Schaefer_Tian.nii.gz'
    input_dir = Path('tvalues')
    output_dir = Path('outputs')  

    # load mask image
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()

    # get t-values
    csv_files = sorted(input_dir.glob('*.csv'))

    if not csv_files:
        raise FileNotFoundError(f'CSV file not found: {input_dir}')

    for csv_path in csv_files:
        values = pd.read_csv(csv_path, usecols=["t-value mean"])["t-value mean"].to_numpy()

        # check if there is 232 ROIs
        if len(values) != 232:
            raise ValueError(
                'the number of ROI should be 232'
            )

        # make new mapping data
        mapped_data = np.zeros_like(mask_data)

        for idx in range(1, 233):
            mapped_data[mask_data == idx] = values[idx - 1]

        output_path = output_dir / f'{csv_path.stem}.nii.gz'

        # make new NifTi image
        new_img = nib.Nifti1Image(
            mapped_data,
            affine=mask_img.affine,
            header=mask_img.header
        )

        # save
        nib.save(new_img, str(output_path))
        print(f'saved: {output_path}')

    print('all csv file converted')


if __name__ == "__main__":
    main()