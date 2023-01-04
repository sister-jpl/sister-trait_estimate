import argparse
import json
import ray
import numpy as np
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
    _ = ray.get([a.do.remote(apply_trait_model,[json_file,args.out_dir]) for a,json_file in zip(actors,args.models)])

    ray.shutdown()

def apply_trait_model(hy_obj,args):
    '''Apply trait model(s) to image and export to file.

    hy_obj = ht.HyTools()
    hy_obj.read_file("/Users/achlus/data1/temp/SISTER_PRISMA_20200216T185549_L2A_CORFL_000/SISTER_PRISMA_20200216T185549_L2A_CORFL_000")
    json_file =  '/Users/achlus/Dropbox/rs/sister/repos/sister-trait_estimate/models/PLSR_raw_coef_LMA_1000_2400.json'
    fc_file = "/Users/achlus/data1/temp/SISTER_PRISMA_20200216T185549_L2A_CORFL_000/SISTER_PRISMA_20200216T185549_L2A_FRCOV_000"

    '''

    json_file,output_dir =args

    with open(json_file, 'r') as json_obj:
        trait_model = json.load(json_obj)
        coeffs = np.array(trait_model['model']['coefficients']).T
        intercept = np.array(trait_model['model']['intercepts'])
        model_waves = np.array(trait_model['wavelengths'])

    output_base = '_'.join(hy_obj.base_name.split('_')[:-1]) + '_' +trait_model["name"].lower()
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

    # Build trait image file
    header_dict = hy_obj.get_header()
    header_dict['wavelength'] = []
    header_dict['data ignore value'] = -9999
    header_dict['data type'] = 4
    header_dict['band names'] = ["%s_mean" % trait_model["name"].lower(),
                                 "%s_std" % trait_model["name"].lower(),
                                 "%s_qa" % trait_model["name"].lower()]
    header_dict['bands'] = len(header_dict['band names'])

    output_name = "%s/%s" % (output_dir,output_base)

    writer = WriteENVI(output_name,header_dict)

    iterator = hy_obj.iterate(by = 'line',
                  resample=resample)

    trait_est = np.zeros((hy_obj.columns,
                          header_dict['bands']))

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
        trait_est[:,0] = trait_pred.mean(axis=1)
        trait_est[:,1] = trait_pred.std(ddof=1,axis=1)
        qa = (trait_est[:,0] > trait_model['model_diagnostics']['min']) & (trait_est[:,0] < trait_model['model_diagnostics']['max'])
        trait_est[:,2] = qa.astype(int)
        nd_mask = hy_obj.mask['no_data'][iterator.current_line]
        trait_est[~nd_mask] = -9999

        writer.write_line(trait_est, iterator.current_line)


    writer.close()


if __name__== "__main__":
    main()
