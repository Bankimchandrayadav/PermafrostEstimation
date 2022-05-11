# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # About
# ### This notebook clips all rasters to study area's extent and sets their 'no data' values.
# %% [markdown]
# # [1](1) Libraries

# %%
import numpy as np, matplotlib.pyplot as plt, pandas as pd, geopandas as gpd, fiona, glob, rasterio as rio, time
from osgeo import gdal
from rasterio import mask as mask
from rasterio.plot import show
from tqdm.notebook import tqdm as td
start = time.time()

# %% [markdown]
# # [2](2) Read shapefile and change crs

# %%
shpInName = "../07_Data/02_Vector/01_Study_Area/tsokar.shp"  # input shapefile name
shpIn = gpd.read_file(shpInName)  # input file read
shpOutName = "../07_Data/02_Vector/01_Study_Area/tsokar_reproj.shp"  # out file name
shpIn.to_crs(epsg=32643, inplace=True)  # set the crs to raster files' crs
shpIn.to_file(shpOutName)  # new shapefile saved

# %% [markdown]
# # [3](3) Clip rasters to study area's extent

# %%
shpOut = fiona.open(shpOutName, 'r')  # [1] read shapefile
shapes = [feature["geometry"] for feature in shpOut]

rasters = glob.glob("../07_Data/01_Raster/01_Xy_Layers/*.tif")  # [2] get list of input rasters 

for i in td(range(len(rasters)), desc='Clipping'):  # [2] clip and save rasters
    with rio.open(rasters[i]) as src:  
        out_image, out_transform = mask.mask(src, shapes, crop=True)  # clip
        out_meta = src.meta  # profile of input raster 
    out_meta.update({"driver": "GTiff", "height": 1428, "width": 1445, "transform": out_transform})  # update profile
    outName = rasters[i].split('\\')[0] + "/01_Reproj/" + rasters[i].split('\\')[1]  # name of out file
    with rio.open(outName, "w", **out_meta) as dest:  # save clipped raster with updated profile
        dest.write(out_image)

# %% [markdown]
# # [4](4) Set the same no data of all rasters

# %%
rasters = glob.glob("../07_Data/01_Raster/01_Xy_Layers/01_Reproj/*.tif")  # [1] get list of rasters

for i in td(range(len(rasters)), desc='NoDataSetting'):  # [2] set no data: get input rasters info
    with rio.open(rasters[i]) as src:
        out_image = src.read(1)  # read as array
        out_meta = src.meta  # profile of input raster
        nodata = out_meta['nodata']  # no data value of input raster
    
    out_image = out_image.astype('float32')  # [3] set no data: convert raster array to float  
    out_image[out_image==nodata] = np.nan  # replace original 'no data' values by np.nan in the array
    out_meta.update({"dtype":"float32","nodata":np.nan})  # update data type and no data info in the profile also
    
    outName = rasters[i].split('01_Reproj\\')[0] + "02_NoData_Set/" + rasters[i].split('\\')[1]  # [4] save file: out file name
    with rio.open(outName, 'w', **out_meta) as dest:  # save file with updated profile info
        dest.write(out_image, 1)  

# %% [markdown]
# # [5](5) Plots of new rasters

# %%
rastersFinal = glob.glob("../07_Data/01_Raster/01_Xy_Layers/02_NoData_Set/*.tif")  # [1] list of files

Names = ['PERMAFROST BASE LAYER', 'DEM', 'EMISSIVITY', 'LST', 'PISR', 'WETNESS INDEX', ]  # [2] plot rasters: names for figure titles
for i in td(range(len(rastersFinal)), desc='Saving Images'):
    src=rio.open(rastersFinal[i])
    plt.figure(dpi=300)
    plt.imshow(rio.open(rastersFinal[i]).read(1), cmap='binary')
    plt.title(Names[i])
    plt.colorbar(extend='both')
    plt.xlabel='Grid columns'
    plt.xlabel='Grid rows'
    plt.savefig("../05_Images/01_X_Files/{}_{}".format(str(i+1).zfill(2), Names[i]), bbox_inches='tight', facecolor='w')
    plt.close()
print('Time elapsed: ', np.round(time.time()-start,2), 'secs')


