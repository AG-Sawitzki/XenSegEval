import cv2
import geopandas as gpd



gdf = gpd.read_file('/data/cephfs-1/work/groups/sawitzki/users/juno12_c/segmentation/labels/roi.geojson')
xx, yy = gdf.iloc[0]['geometry'].exterior.xy

x = np.array(xx)
y = np.array(yy)

x.shape = (len(x),1)
y.shape = (len(y),1)

xy = np.hstack((x,y))

mm = cv2.moments(xy)