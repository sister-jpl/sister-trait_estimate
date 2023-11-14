import glob
import json
import os
import sys
import shutil
import ray
import numpy as np
from osgeo import gdal
import hytools as ht
from PIL import Image
import matplotlib.pyplot as plt

def main():

    pge_path = os.path.dirname(os.path.realpath(__file__))

    run_config_json = sys.argv[1]

    with open(run_config_json, 'r') as in_file:
        run_config =json.load(in_file)

    experimental = run_config['inputs']['experimental']
    if experimental:
        disclaimer = "(DISCLAIMER: THIS DATA IS EXPERIMENTAL AND NOT INTENDED FOR SCIENTIFIC USE) "
    else:
        disclaimer = ""

    os.mkdir('output')
    os.mkdir('temp')

    crid = run_config["inputs"]["crid"]

    rfl_base_name = os.path.basename(run_config['inputs']['reflectance_dataset'])
    sister,sensor,level,product,datetime,in_crid = rfl_base_name.split('_')

    rfl_file = f'input/{rfl_base_name}/{rfl_base_name}.bin'
    rfl_met = rfl_file.replace('.bin','.met.json')
    fc_base_name = os.path.basename(run_config['inputs']['frcov_dataset'])
    fc_file = f'input/{fc_base_name}/{fc_base_name}.tif'

    qlook_file = f'output/SISTER_{sensor}_L2B_VEGBIOCHEM_{datetime}_{crid}.png'
    qlook_met = qlook_file.replace('.png','.met.json')

    models = glob.glob(f'{pge_path}/models/PLSR*.json')

    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus = len(models))

    HyTools = ray.remote(ht.HyTools)
    actors = [HyTools.remote() for rfl_file in models]

    # Load data
    _ = ray.get([a.read_file.remote(rfl_file,'envi') for a,b in zip(actors,models)])

    # Set fractional cover mask
    fc_obj = gdal.Open(fc_file)
    veg_mask = fc_obj.GetRasterBand(2).ReadAsArray() >= run_config['inputs']['veg_cover']

    _ = ray.get([a.set_mask.remote(veg_mask,'veg') for a,b in zip(actors,models)])

    _ = ray.get([a.do.remote(apply_trait_model,[json_file,crid]) for a,json_file in zip(actors,models)])

    ray.shutdown()

    bands = []

    if sensor != 'DESIS':
        for trait_abbrv in ['NIT','CHL','LMA']:
            tif_file = f'output/SISTER_{sensor}_L2B_VEGBIOCHEM_{datetime}_{crid}_{trait_abbrv}.tif'
            gdal_obj = gdal.Open(tif_file)
            band = gdal_obj.GetRasterBand(1)
            band_arr = np.copy(band.ReadAsArray())
            bands.append(band_arr)

        rgb=  np.array(bands)
        rgb[rgb == band.GetNoDataValue()] = np.nan

        rgb = np.moveaxis(rgb,0,-1).astype(float)
        bottom = np.nanpercentile(rgb,5,axis = (0,1))
        top = np.nanpercentile(rgb,95,axis = (0,1))
        rgb = np.clip(rgb,bottom,top)
        rgb = (rgb-np.nanmin(rgb,axis=(0,1)))/(np.nanmax(rgb,axis= (0,1))-np.nanmin(rgb,axis= (0,1)))
        rgb = (rgb*255).astype(np.uint8)
        im = Image.fromarray(rgb)
        description = f'{disclaimer}Vegetation biochemistry RGB quicklook. R: Nitrogen, G: Chlorophyll, B: Leaf Mass ' \
                      f'per Area'

    else:

        tif_file = f'output/SISTER_{sensor}_L2B_VEGBIOCHEM_{datetime}_{crid}_CHL.tif'
        gdal_obj = gdal.Open(tif_file)
        band = gdal_obj.GetRasterBand(1)
        band_arr = np.copy(band.ReadAsArray())
        band_arr[band_arr == band.GetNoDataValue()] = np.nan

        bottom = np.nanpercentile(band_arr,5)
        top = np.nanpercentile(band_arr,95)
        band_arr = np.clip(band_arr,bottom,top)
        band_arr = (band_arr-np.nanmin(band_arr))/(np.nanmax(band_arr)-np.nanmin(band_arr))

        cmap = plt.get_cmap('RdYlGn')
        qlook = cmap(band_arr)[:,:,:3]
        qlook = (255 * qlook).astype(np.uint8)
        qlook[band_arr == -9999] = 0

        im = Image.fromarray(qlook, 'RGB')
        description = f'{disclaimer}Vegetation biochemistry quicklook. Chlorophyll'

    im.save(qlook_file)

    generate_metadata(rfl_met,
                      qlook_met,
                      {'product': 'VEGBIOCHEM',
                      'processing_level': 'L2B',
                      'description' : description,
                       })

    shutil.copyfile(run_config_json,
                    qlook_file.replace('.png','.runconfig.json'))

    if os.path.exists("run.log"):
        shutil.copyfile('run.log',
                        qlook_file.replace('.png','.log'))

    # If experimental, prefix filenames with "EXPERIMENTAL-"
    if experimental:
        for file in glob.glob(f"output/SISTER*"):
            shutil.move(file, f"output/EXPERIMENTAL-{os.path.basename(file)}")


def generate_metadata(in_file,out_file,metadata):

    with open(in_file, 'r') as in_obj:
        in_met =json.load(in_obj)

    for key,value in metadata.items():
        in_met[key] = value

    with open(out_file, 'w') as out_obj:
        json.dump(in_met,out_obj,indent=3)


def apply_trait_model(hy_obj,args):
    '''Apply trait model(s) to image and export to file.

    '''

    json_file,crid =args

    with open(json_file, 'r') as json_obj:
        trait_model = json.load(json_obj)
        coeffs = np.array(trait_model['model']['coefficients']).T
        intercept = np.array(trait_model['model']['intercepts'])
        model_waves = np.array(trait_model['wavelengths'])

    if (hy_obj.wavelengths.min() > model_waves.min()) |  (hy_obj.wavelengths.max() < model_waves.max()):
        print('%s model wavelengths outside of image wavelength range, skipping....' % trait_model["full_name"])
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

    trait_abbrv = trait_model["short_name"].upper()
    sister,sensor,level,product,datetime,in_crid =  hy_obj.base_name.split('_')

    temp_file =  f'temp/SISTER_{sensor}_L2B_VEGBIOCHEM_{datetime}_{crid}_{trait_abbrv}.tif'
    out_file =  temp_file.replace('temp','output')

    band_names = ["%s_mean" % trait_model["short_name"].lower(),
                                 "%s_std_dev" % trait_model["short_name"].lower(),
                                 "%s_qa_mask" % trait_model["short_name"].lower()]

    units= [trait_model["full_units"].upper(),
            trait_model["full_units"].upper(),
            "NA"]

    descriptions= ["%s MEAN" % trait_model["full_name"].upper(),
                  "%s STANDARD DEVIATION" % trait_model["full_name"].upper(),
                  "QUALITY ASSURANCE MASK"]


    in_file = gdal.Open(hy_obj.file_name)

    # Set the output raster transform and projection properties
    driver = gdal.GetDriverByName("GTIFF")
    tiff = driver.Create(temp_file,
                         hy_obj.columns,
                         hy_obj.lines,
                         3,
                         gdal.GDT_Float32)

    tiff.SetGeoTransform(in_file.GetGeoTransform())
    tiff.SetProjection(in_file.GetProjection())
    tiff.SetMetadataItem("DESCRIPTION","L2B VEGETATION BIOCHEMISTRY %s" % trait_model["full_name"].upper() )

    # Write bands to file
    for i,band_name in enumerate(band_names,start=1):
        band = tiff.GetRasterBand(i)
        band.WriteArray(trait_array[i-1])
        band.SetDescription(band_name)
        band.SetNoDataValue(hy_obj.no_data)
        band.SetMetadataItem("UNITS",units[i-1])
        band.SetMetadataItem("DESCRIPTION",descriptions[i-1])
    del tiff, driver

    os.system(f"gdaladdo -minsize 900 {temp_file}")
    os.system(f"gdal_translate {temp_file} {out_file} -co COMPRESS=LZW -co TILED=YES -co COPY_SRC_OVERVIEWS=YES")

    rfl_met = hy_obj.file_name.replace('.bin','.met.json')
    trait_met =out_file.replace('.tif','.met.json')

    generate_metadata(rfl_met,
                      trait_met,
                      {'product': 'VEGBIOCHEM',
                      'processing_level': 'L2B',
                      'description' : trait_model["full_name"],
                       })


if __name__== "__main__":
    main()
