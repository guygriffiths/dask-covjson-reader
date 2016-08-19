'''
Created on 19 Aug 2016

@author: Guy Griffiths
'''

import requests
import json
import numpy as np
import itertools
import dask.array as da

def get_data(location):
    """ Retrieves data from a URL """
    # This is factored out so we can use a different retrieval method if required.
    # Originally used urllib2, but it had SSL issues on my machine
    response = requests.get(location)
    return response.content

def get_dask_arrays(location):
    """ Parses a tiled CoverageJSON document and returns a set of dask arrays.
    
    The return object is a dictionary whose keys are of the form:
    
    <parameter name>-<tiled dimensions>_tiling
    
    or
    
    <parameter name>-untiled
    
    in the case where the tile set contains all the data.  For example, the CoverageJSON document at:
    https://covjson.org/playground/coverages/grid-tiled.covjson
    
    returns an object with the keys:
    
    FOO-yx_tiling
    FOO-t_tiling
    FOO-untiled
    
    The values of the return dictionary are the dask arrays.
    
    The location argument takes the URL of the CoverageJSON document.
    """
    
    # We will return a dictionary of dask Arrays.
    # It will contain one dask Array for each parameter and tileset
    dask_arrays = {}
    
    # First, load the CoverageJSON and parse it
    coverage = json.loads(get_data(location))
    
    # In a fully-fledged application, we would probably want to process the
    # rest of the coverage here to create the domain(s), etc.
    
    # Find all of the named parameters
    parameters = coverage['parameters'].keys()
    
    for param in parameters:
        # The range information for the current parameter
        p_range = coverage['ranges'][param]
        # The axis names for the current parameter
        p_axis_names = p_range['axisNames']
        # The shape of the data for the current parameter
        p_shape = p_range['shape']
        # The available tilesets for the current parameter
        p_tilesets = p_range['tileSets']
        
        for p_tileset in p_tilesets:
            # The shape of the tiles in this tileset
            tile_shape = p_tileset['tileShape']
            
            # Find out which axes this tileset is using to split tiling
            # This will be used as a key to the dictionary of dask arrays we return
            axes_str = ""
            for axis_tile_size, axis_name in zip(tile_shape, p_axis_names):
                if axis_tile_size is not None:
                    axes_str += axis_name
            if(len(axes_str) == 0):
                # We could use this check to avoid creating a dask array.
                # After all, their main aim is to deal with tiled data.
                da_key = param + "-untiled"
            else:
                da_key = param + "-" + axes_str + "_tiling"
    
            # Find out how each of the axes will be split up.
            # We use the tile shape and the overall axis size to determine this
            
            # This will store the tile sizes used along each axis.
            # For example, for a 2d 10x11 system split into tiles of 3x3 except at the edges, it would be:
            # [[3,3,3,1],[3,3,3,2]]
            # This (converted to tuples) is the object we pass to da.Array to specify the tiling of the dask array 
            axes_tile_sizes = []
            # This is the valid coordinates of the axes.  For example, for a 3x4x5 system, it will be:
            # [[0,1,2],[0,1,2,3],[0,1,2,3,4]]
            # We will use it to generate all of the possible tile coordinates.  That's needed to create the 
            # dictionary we pass to da.Array, so that dask knows how to retrieve each tile
            axes_coords = []
            for axis_size, axis_tile_size in zip(p_shape, tile_shape):
                if(axis_tile_size is None):
                    # This implies the entire axis is available in each tile
                    axis_tile_size = axis_size
                # Number of fully-sized tiles along this axis
                n_full_tiles = axis_size / axis_tile_size
                # The size of the final tile along this axis.  If this is zero, there isn't one.
                final_tile = axis_size % axis_tile_size
                
                axis_tiles = n_full_tiles * [axis_tile_size]
                if(final_tile > 0):
                    axis_tiles = axis_tiles + [final_tile]

                axes_tile_sizes.append(tuple(axis_tiles))
                axes_coords.append(range(len(axis_tiles)))
            
            # Now construct the dask graph dictionary
            dask_graph = {}
            # This finds all of the combinations of the axis indices
            tile_coords = itertools.product(*axes_coords)
            for coord in tile_coords:
                # Each entry in the dask graph needs:
                #     The parameter name and tile coordinates as the key
                #     A value which is a tuple containing:
                #        The function used to load the tile
                #        The arguments to that function
                dask_graph[(param,) + coord] = (get_tile, p_tileset['urlTemplate'], p_axis_names, coord)
            
            # Now create the dask Array for this parameter + tileset, and add it to the return dictionary
            dask_arrays[da_key] = da.Array(dask_graph, param, tuple(axes_tile_sizes))
    return dask_arrays 
    
def get_tile(url_template, axis_names, tile_indices):
    """ Gets specified tile data.  Takes parameters:
            url_template: This is the URL to retrieve a tile from.  All occurrences of {axisName} are replaced with the correct tile index.
            axis_names: The axis names to be replaced with the specified indices.
            tile_indices: The index to retrieve along each axis.
        Example:
        
        get_tile('http://localhost/tiles/{x}/{y}/{z}', ['x', 'y', 'z'], [0, 1, 2])
        
        will fetch the data at the URL: http://localhost/tiles/0/1/2
        
        It is perfectly permissible (and a common use-case) to specify axis names which are not replaced.  For example:
        
        get_tile('http://localhost/tiles/{x}/{y}', ['x', 'y', 'z'], [0, 1, 2])
        
        will fetch the data at the URL: http://localhost/tiles/0/1
        
        This assumes that the data is in CoverageJSON format, and does the work of fetching the data, parsing it, and extracting the actual
        data as a numpy array.
    """
    for axis, tile_index in zip(axis_names, tile_indices):
        url_template = url_template.replace('{' + axis + '}', str(tile_index))
    tile_data = json.loads(get_data(url_template))
    tile_values = np.array(tile_data['values']).reshape(tile_data['shape'])
    return tile_values

if __name__ == '__main__':
    # Usage example.
    arrs = get_dask_arrays('https://covjson.org/playground/coverages/grid-tiled.covjson')
    
    for name, data in arrs.iteritems():
        print name+':'
        # Here we just convert it to a numpy array and print it to screen.
        # To make use of dask we probably want to do something more useful, like performing calculations on it, or slicing it.
        print np.array(data)
        print 