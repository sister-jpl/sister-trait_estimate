# SISTER Trait estimate PGE Documentation

## Description

The L2B trait estimate PGE takes as input a surface reflectance dataset and a fractional cover map and applies partial least squares regression (PLSR) algorithms to generate maps of the following vegetation canopy traits:

- Chlorophyll content (ug/cm2) 
- Nitrogen concentration (mg/g)
- Leaf mass per area (g/m2)

Permuted PLSR models were developed using coincident NEON AOP canopy spectra, downsampled to 10nm, and field data collected by Wang et al. (2020). In addition to biochemical trait estimates, per-pixel uncertainties are also calculated as well as a quality assurance mask which flags pixels with trait estimates outside of the range of data used to build the model. For DESIS imagery only chlorophyll is estimated, models for nitrogen and leaf mass per area require
infrared wavelengths outside of the spectral range of DESIS.

### AVIRIS Classic vegetation trait quicklook

![AVIRIS trait estimate](./trait_estimate_example.png)

### References

- Wang, Z., Chlus, A., Geygan, R., Ye, Z., Zheng, T., Singh, A., Couture, J.J., Cavender‐Bares, J., Kruger, E.L. and Townsend, P.A., 2020. Foliar functional traits from imaging spectroscopy across biomes in eastern North America. New Phytologist, 228(2), pp.494-511.


## PGE Arguments

The L2B trait estimate PGE  takes the following argument(s):

| Argument            | Description                                           | Default |
|---------------------|-------------------------------------------------------|---------|
| reflectance_dataset | L2A reflectance dataset                               | -       |
| frcover_dataset     | L2B fractional cover dataset                          | -       |
| veg_cover           | Minimum vegetation cover to apply algorithm (0.0-1.0) | 0.5     |
| crid                | Composite release identifier                          | '000'   |
| experimental        | Designates outputs as "experimental"                  | 'True'  |

## Outputs

The outputs of the L2B vegetation trait estimate PGE use the following naming convention:

    (EXPERIMENTAL-)SISTER_<SENSOR>_L2B_VEGBIOCHEM_<YYYYMMDDTHHMMSS>_<CRID>_<SUBPRODUCT>

Note that the "EXPERIMENTAL-" prefix is optional and is only added when the "experimental" flag is set to True.

The following data products are produced:

| Product description                          | Units  | Example filename                                                   |
|----------------------------------------------|--------|--------------------------------------------------------------------|
| Chlorophyll COGeotiff                        | -      | SISTER\_AVCL\_L2B\_VEGBIOCHEM\_20130612T175359\_000\_CHL.tif       |
| 1. Chlorophyll content                       | ug/cm2 |                                                                    |
| 2. Chlorophyll content uncertainty           | ug/cm2 |                                                                    |
| 3. Quality assurance mask                    | -      |                                                                    |
| Chlorophyll metadata (STAC formatted)        | -      | SISTER\_AVCL\_L2B\_VEGBIOCHEM\_20130612T175359\_000\_CHL.json      |
| Nitrogen COGeotiff                           | mg/g   | SISTER\_AVCL\_L2B\_VEGBIOCHEM\_20130612T175359\_000\_NIT.tif       |
| 1. Nitrogen concentration                    | -      |                                                                    |
| 2. Nitrogen concentration uncertainty        | mg/g   |                                                                    |
| 3. Quality assurance mask                    | -      |                                                                    |
| Nitrogen metadata (STAC formatted)           | -      | SISTER\_AVCL\_L2B\_VEGBIOCHEM\_20130612T175359\_000\_NIT.json      |
| Leaf mass per area COGeotiff                 | -      | SISTER\_AVCL\_L2B\_VEGBIOCHEM\_20130612T175359\_000\_LMA.tif       |
| 1. Leaf mass per area                        | g/m2   |                                                                    |
| 2. Leaf mass per area uncertainty            | g/m2   |                                                                    |
| 3. Quality assurance mask                    | -      |                                                                    |
| Leaf mass per area metadata (STAC formatted) | -      | SISTER\_AVCL\_L2B\_VEGBIOCHEM\_20130612T175359\_000\_LMA.json      |
| Quicklook                                    | -      | SISTER\_AVCL\_L2A\_VEGBIOCHEM\_20130612T175359\_000.png            |
| PGE runconfig                                | -      | SISTER\_AVCL\_L2B\_VEGBIOCHEM\_20130612T175359\_000.runconfig.json |
| PGE log                                      | -      | SISTER\_AVCL\_L2B\_VEGBIOCHEM\_20130612T175359\_000.log            |

Metadata files are [STAC formatted](https://stacspec.org/en) and compatible with tools in the [STAC ecosystem](https://stacindex.org/ecosystem).

## Executing the Algorithm

This algorithm requires [Anaconda Python](https://www.anaconda.com/download)

To install and run the code, first clone the repository and execute the install script:

    git clone https://github.com/sister-jpl/sister-trait_estimate.git
    cd sister-trait_estimate
    ./install.sh
    cd ..

Then, create a working directory and enter it:

    mkdir WORK_DIR
    cd WORK_DIR

Copy input files to the work directory. For each "dataset" input, create a folder with the dataset name, then download 
the data file(s) and STAC JSON file into the folder.  For example, the reflectance dataset input would look like this:

    WORK_DIR/SISTER_AVCL_L2A_CORFL_20130612T175359_000/SISTER_AVCL_L2A_CORFL_20130612T175359_000.bin
    WORK_DIR/SISTER_AVCL_L2A_CORFL_20130612T175359_000/SISTER_AVCL_L2A_CORFL_20130612T175359_000.hdr
    WORK_DIR/SISTER_AVCL_L2A_CORFL_20130612T175359_000/SISTER_AVCL_L2A_CORFL_20130612T175359_000.json

Finally, run the code 

    ../sister-trait_estimate/pge_run.sh --reflectance_dataset SISTER_AVCL_L2A_CORFL_20130612T175359_000 --frcov_dataset SISTER_AVCL_L2B_FRCOV_20130612T175359_000
