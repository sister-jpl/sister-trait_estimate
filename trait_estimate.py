import argparse
import json
import os
import ray
import numpy as np
from osgeo import gdal
import hytools as ht
from hytools.io.envi import WriteENVI


def main():
    desc = "Estimate vegetation functional traits"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('rfl_file', type=str,
                        help='Input reflectance image')
    parser.add_argument('frac_file', type=str,
                        help='Input fractional cover image')
    parser.add_argument('out_dir', type=str,
                        help='Output directory')
    parser.add_argument('--models', nargs='+',
                        help='Trait models', required=True)

    args = parser.parse_args()

    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus = len(args.models))

    HyTools = ray.remote(ht.HyTools)
    actors = [HyTools.remote() for rfl_file in args.models]

    # Load data
    _ = ray.get([a.read_file.remote(args.rfl_file,'envi') for a,b in zip(actors,args.models)])

    # Set fractional cover mask
    fc_obj = ht.HyTools()
    fc_obj.read_file(fc_file)
    _ = ray.get([a.set_mask.remote(fc_obj.get_band(1) >= .5,'veg') for a,b in zip(actors,args.models)])

    del fc_obj

    _ = ray.get([a.do.remote(apply_trait_model,[json_file,args.out_dir]) for a,json_file in zip(actors,args.models)])

    ray.shutdown()



def apply_trait_model(hy_obj,args):
    '''Apply trait model(s) to image and export to file.

    hy_obj = ht.HyTools()
    hy_obj.read_file("/Users/achlus/data1/temp/SISTER_PRISMA_20200216T185549_L2A_CORFL_000/SISTER_PRISMA_20200216T185549_L2A_CORFL_000")
    json_file =  '/Users/achlus/Dropbox/rs/sister/repos/sister-trait_estimate/models/PLSR_raw_coef_LMA_1000_2400.json'
    fc_file = "/Users/achlus/data1/temp/SISTER_PRISMA_20200216T185549_L2A_CORFL_000/SISTER_PRISMA_20200216T185549_L2A_FRCOV_000"
    fc_obj = ht.HyTools()
    fc_obj.read_file(fc_file)

    hy_obj.mask['veg'] = fc_obj.get_band(1) >= .5
    output_dir = '/Users/achlus/data1/temp/'

    '''

    json_file,output_dir =args

    with open(json_file, 'r') as json_obj:
        trait_model = json.load(json_obj)
        coeffs = np.array(trait_model['model']['coefficients']).T
        intercept = np.array(trait_model['model']['intercepts'])
        model_waves = np.array(trait_model['wavelengths'])


    output_base = '_'.join(hy_obj.base_name.split('_')[:-2]) + '_TE_' +trait_model["short_name"].upper() + '_CRID'
    print(output_base)

    if (hy_obj.wavelengths.min() > model_waves.min()) |  (hy_obj.wavelengths.max() < model_waves.max()):
        print('%s model wavelengths outside of image wavelength range, skipping....' % trait_model["name"])
        return

    hy_obj.create_bad_bands([[300,400],[1337,1430],[1800,1960],[2450,2600]])
    hy_obj.resampler['type'] = 'cubic'

    #Check if wavelengths match
    resample = not all(x in hy_obj.wavelengths for x in model_waves)
    if resample:
        print('Spectral resampling required')
        hy_obj.resampler['out_waves'] = model_waves
    else:
        wave_mask = [np.argwhere(x==hy_obj.wavelengths)[0][0] for x in model_waves]

    iterator = hy_obj.iterate(by = 'line',
                  resample=resample)

    trait_array = np.zeros((3,hy_obj.lines,
                            hy_obj.columns))

    while not iterator.complete:
        chunk = iterator.read_next()
        if not resample:
            chunk = chunk[:,wave_mask]

        # Apply spectrum transforms
        for transform in  trait_model['model']["transform"]:
            if  transform== "vector":
                norm = np.linalg.norm(chunk,axis=1)
                chunk = chunk/norm[:,np.newaxis]
            if transform == "absorb":
                chunk = np.log(1/chunk)
            if transform == "mean":
                mean = chunk.mean(axis=1)
                chunk = chunk/mean[:,np.newaxis]

        trait_pred = np.dot(chunk,coeffs)
        trait_pred = trait_pred + intercept
        trait_mean = trait_pred.mean(axis=1)
        qa = (trait_mean > trait_model['model_diagnostics']['min']) & (trait_mean < trait_model['model_diagnostics']['max'])

        trait_array[0,iterator.current_line,:] = trait_mean
        trait_array[1,iterator.current_line,:] = trait_pred.std(ddof=1,axis=1)
        trait_array[2,iterator.current_line,:] = qa.astype(int)

        nd_mask = hy_obj.mask['no_data'][iterator.current_line] & hy_obj.mask['veg'][iterator.current_line]
        trait_array[:,iterator.current_line,~nd_mask] = -9999

    geotiff =  "temp/%s.tif" % (output_dir,output_base)

    in_file = gdal.Open(hy_obj.file_name)

    band_names = ["%s_mean" % trait_model["short_name"].lower(),
                                 "%s_std_dev" % trait_model["short_name"].lower(),
                                 "%s_qa_mask" % trait_model["short_name"].lower()]

    units= [trait_model["full_units"].upper(),
            trait_model["full_units"].upper(),
            "NA"]

    descriptions= ["%s MEAN" % trait_model["full_name"].upper(),
                  "%s STANDARD DEVIATION" % trait_model["full_name"].upper(),
                  "QUALITY ASSURANCE MASK"]

    # Set the output raster transform and projection properties
    driver = gdal.GetDriverByName("GTIFF")
    tiff = driver.Create(geotiff,
                         hy_obj.columns,
                         hy_obj.lines,
                         3,
                         gdal.GDT_Float32)

    tiff.SetGeoTransform(in_file.GetGeoTransform())
    tiff.SetProjection(in_file.GetProjection())
    tiff.SetMetadataItem("DESCRIPTION","CANOPY BIOCHEMISTRY PLSR %s" % trait_model["full_name"].upper() )

    # Write bands to file
    for i,band_name in enumerate(band_names,start=1):
        band = tiff.GetRasterBand(i)
        band.WriteArray(trait_array[i-1])
        band.SetDescription(band_name)
        band.SetNoDataValue(hy_obj.no_data)
        band.SetMetadataItem("UNITS",units[i-1])
        band.SetMetadataItem("DESCRIPTION",descriptions[i-1])
    del tiff, driver

    COGtiff =  "output/%s.tif" % (output_dir,output_base)

    os.system("gdal_translate %s %s -of COG -co COMPRESS=LZW" % (geotiff,COGtiff))


if __name__== "__main__":
    main()
