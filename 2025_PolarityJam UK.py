# Original code and installation instructions by Jan Philipp Albrecht, Wolfgang Giese
# https://github.com/polarityjam/polarityjam
# Adapted by U. Kałucka
# Python version used for analysis: 3.8.20

# must run in an environment where polarityJaM is installed
from polarityjam import Extractor, Plotter, PropertiesCollection
from polarityjam import RuntimeParameter, PlotParameter, SegmentationParameter, ImageParameter, SegmentationMode
from polarityjam import PolarityJamLogger
from polarityjam import load_segmenter
from polarityjam.utils.io import read_image

from pathlib import Path

# Setup a logger to only report WARNINGS, Put "INFO" or "DEBUG" to get more information
plog = PolarityJamLogger("WARNING")

### ADAPT ME ###
path_root = Path("")
input_file = path_root.joinpath("X:input location")
output_path = path_root.joinpath("X:output location")
output_file_prefix = "Prefix"
### ADAPT ME ###

# read input
img = read_image(input_file)

# describe your image with ImageParameter
params_image = ImageParameter()

# set the channels
params_image.channel_organelle = -1  # here no golgi channel
params_image.channel_nucleus = 0 # DAPI channel
params_image.channel_junction = 1 # ZO-1 channel
params_image.channel_expression_marker = 1 # ZO-1 channel
params_image.pixel_to_micron_ratio = 1.5385 # insert correct value

print(params_image)

# define other parameters, use default values
params_runtime = RuntimeParameter()
params_plot = PlotParameter()

# Plot info
print(params_runtime)
print(params_plot)

# EXAMPLE: change some parameters
params_runtime.membrane_thickness = 6
params_runtime.estimated_cell_diameter = 300

# Print which algorithm is used
print("Used algorithm for segmentation: %s " % params_runtime.segmentation_algorithm)

# Now define your segmenter and segment your image with the default algorithm and default parameters.
cellpose_segmentation, _ = load_segmenter(params_runtime)

# prepare your image for segmentation
img_prepared, img_prepared_params = cellpose_segmentation.prepare(img, params_image)

# Define a plotter and check your for segmentation prepared image
plotter = Plotter(params_plot)

# plot input
plotter.plot_channels(img_prepared, img_prepared_params, output_path, input_file);

# now segment your prepared image to get the masks
mask = cellpose_segmentation.segment(img_prepared, input_file)

# plot segmentation mask to check the quality
plotter.plot_mask(mask, img_prepared, img_prepared_params, output_path, output_file_prefix);

import numpy as np

# create a dummy mask
masks = np.zeros((100, 100))

np.save("X:output location", {"masks": masks})

# plot the cell orientation of a specific image in your collection
# plotter.plot_shape_orientation(collection, "K1");  # image is automatically saved in output_path

# feature extraction
collection = PropertiesCollection()
extractor = Extractor(params_runtime)
extractor.extract(img, params_image, mask, output_file_prefix, output_path, collection)

collection.dataset.head()
collection.dataset.to_csv(output_path.joinpath('features.csv'))

# or simply plot the whole collection
plotter.plot_collection(collection);

# Output omitted, PolarityWeb app used for visualisation
