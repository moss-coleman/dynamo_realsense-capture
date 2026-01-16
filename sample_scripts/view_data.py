from dynamo import view_open3d

# data_location = "../data/"
folder = "../data/"
file = "../data/"
full = 1 #set to 1 to view all frames, set to 0 to view every 10 frames

# view_open3d.viewPointClouds(folder,full)
# view_open3d.viewPointClouds(folder,True)

view_open3d.viewSinglePC(folder,False)
