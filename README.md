# peak_roi
Peak picking using ROI method
It has 3 script files.

Each file can be run separately.


# parserxml.py
parses the mzxml file to generate the raw data. The multiprocessing modules can process mutiple file in a folder at once.


# roinew2.py

it creates roi-seed data by clustering the mz values in a bin of thresold mass value (0.004) from the first scan.
Thereafter, it takes data from the rest of scans (scan-wise) to be added into the bins of roi-seed if within the thresold otherwise add a new roi into the roi-seed.This process iterates untill data from all the scan were added to the roi-seed and finally generates the roi values.

All the roi were filtered based on the criteria of having minimum number of points in the roi list, having intensity more than thresold values, and should not emcompasses more than 50% of the total scans of data.

# peakfinal2.py

It takes input of roi values, and raw data.
The data corresponding to each roi were retrived.
Interpolation were performed on each roi to generate EIC 
noise estimation and denoising using sg filter
and peak picking were performed using Scipy find peak using prominence and wlen as criteria.
Area were calculated using trapz (intergral function).

