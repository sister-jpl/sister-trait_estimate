import argparse
import json
import ray
import numpy as np
import hytools as ht
from hytools.io.envi import WriteENVI


def main():
    desc = "Estimate traits"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('rfl_file', type=str,
                        help='Input reflectance image')
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
    '''

    json_file,output_dir =args

    with open(json_file, 'r') as json_obj:
        trait_model = json.load(json_obj)
        coeffs = np.array(trait_model['model']['coefficients'])
        intercept = np.array(trait_model['model']['intercepts'])
        model_waves = np.array(trait_model['wavelengths'])

    output_base = hy_obj.base_name.replace('rfl',trait_model["name"].lower())
    print(output_base)

    hy_obj.create_bad_bands([[300,400],[1337,1430],[1800,1960],[2450,2600]])
    hy_obj.resampler['type'] = 'cubic'

    #Check if wavelengths match
    resample = not all(x in hy_obj.wavelengths for x in model_waves)
    if resample:
        hy_obj.resampler['out_waves'] = model_waves
    else:
        wave_mask = [np.argwhere(x==hy_obj.wavelengths)[0][0] for x in model_waves]

    # Build trait image file
    header_dict = hy_obj.get_header()
    header_dict['wavelength'] = []
    header_dict['data ignore value'] = -9999
    header_dict['data type'] = 4
    header_dict['band names'] = ["%s_mean" % trait_model["name"].lower(),
                                 "%s_std" % trait_model["name"].lower()]
    header_dict['bands'] = len(header_dict['band names'])

    output_name = "%s/%s" % (output_dir,output_base)

    writer = WriteENVI(output_name,header_dict)

    iterator = hy_obj.iterate(by = 'chunk',
                  chunk_size = (100,100),
                  resample=resample)

    while not iterator.complete:
        chunk = iterator.read_next()
        if not resample:
            chunk = chunk[:,:,wave_mask]

        trait_est = np.zeros((chunk.shape[0],
                                chunk.shape[1],
                                header_dict['bands']))

        # Apply spectrum transforms
        for transform in  trait_model['model']["transform"]:
            if  transform== "vector":
                norm = np.linalg.norm(chunk,axis=2)
                chunk = chunk/norm[:,:,np.newaxis]
            if transform == "absorb":
                chunk = np.log(1/chunk)
            if transform == "mean":
                mean = chunk.mean(axis=2)
                chunk = chunk/mean[:,:,np.newaxis]

        trait_pred = np.einsum('jkl,ml->jkm',chunk,coeffs, optimize='optimal')
        trait_pred = trait_pred + intercept
        trait_est[:,:,0] = trait_pred.mean(axis=2)
        trait_est[:,:,1] = trait_pred.std(ddof=1,axis=2)

        nd_mask = hy_obj.mask['no_data'][iterator.current_line:iterator.current_line+chunk.shape[0],
                                         iterator.current_column:iterator.current_column+chunk.shape[1]]
        trait_est[~nd_mask] = -9999
        writer.write_chunk(trait_est,
                           iterator.current_line,
                           iterator.current_column)
    writer.close()


if __name__== "__main__":
    main()
