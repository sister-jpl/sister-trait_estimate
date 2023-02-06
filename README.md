# SISTER Trait estimate PGE Documentation## DescriptionThe L2B trait estimate PGE takes as input a surface reflectance dataset and a fraction cover map and applies a partial least squares regression (PLSR) algorithms to generate maps of the following vegetation canopy traits:

- Chlorophyll content (ug/cm2) 
- Nitrogen concentration (g/mg)
- Leaf mass per area (g/m2)

For DESIS imagery only chlorophyll is estimated, models for nitrogen and leaf mass per area required
infrared wavelengths outside of the spectral range of DESIS.
## PGE ArgumentsIn addition to required MAAP job submission arguments the L2A spectral resampling PGE also takes the following argument(s):|Argument| Type |  Description | Default||---|---|---|---|| reflectance_dataset| string |L2A reflectance dataset granule URL| -|| frcover_dataset| string |L2B fractional cover granule URL| -|
| veg_cover| string | Minimum vegetation cover to apply algorith (0.0-1.0)| 0.5|
| crid| string | Composite release identifier| 000|
## OutputsThe L2B trait estimate PGE exports  Cloud Optimized GeoTiffs (COGs). The outputs of the PGE use the following naming convention:	    SISTER_<SENSOR>_L2B_VEGBIOCHEM_<YYYYMMDDTHHMMSS>_CRID_CHL.tif
	    SISTER_<SENSOR>_L2B_VEGBIOCHEM_<YYYYMMDDTHHMMSS>_CRID_LMA.tif
	    SISTER_<SENSOR>_L2B_VEGBIOCHEM_<YYYYMMDDTHHMMSS>_CRID_NIT.tif|Subproduct| Description |  Units |Example filename ||---|---|---|---|| CHL| Chlorophyll content datacube | ug/cm2 | SISTER\_AVNG\_L2B\_VEGBIOCHEM\_20220502T180901\_001\_CHL.tif || | Chlorophyll content uncertainty | ug/cm2 |  |
| | Quality assurance mask  | - |  || NIT| Nitrogen concentration datacube | mg/g| SISTER\_AVNG\_L2B\_VEGBIOCHEM\_20220502T180901\_001\_NIT.tif || | Nitrogen concentration uncertainty | mg/g |  |
| | Quality assurance mask  | - |  |
| LMA| Leaf mass per area datacube | g/m2 | SISTER\_AVNG\_L2B\_VEGBIOCHEM\_20220502T180901\_001\_LMA.tif || | LMA content uncertainty | g/m2 |  |
| | Quality assurance mask  | - |  |
## Algorithm registration

This algorithm can be registered using the algorirthm_config.yml file found in this repository:

	from maap.maap import MAAP
	import IPython
	
	maap = MAAP(maap_host="sister-api.imgspec.org")

	trait_estimate_alg_yaml = './sister-trait_estimate/algorithm_config.yaml'
	maap.register_algorithm_from_yaml_file(file_path= trait_estimate_alg_yaml)

## Example
	vegbiochem_job_response = maap.submitJob(
	                        algo_id="sister-trait_estimate",
	                        version="1.0.0",
	                        reflectance_dataset= '../SISTER_AVNG_L2A_RFL_20220502T180901_001',
	                        frcov_dataset= '../SISTER_AVNG_L2A_FRCOV_20220502T180901_001',
	                        veg_cover = 0.5,
	                        crid = '001'
	                        publish_to_cmr=False,
	                        cmr_metadata={},
	                        queue="sister-job_worker-16gb",
	                        identifier= 'SISTER_AVNG_20170827T175432_L2B_VEGBIOCHEM_001')
	                        
	                        
	                        
                        
                        
                        
                        
                        