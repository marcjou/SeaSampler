import seasampler as samp

#object = samp.SeaSampler('/home/marc/Projects/Mednet/tMednet/src/input_files/SeaSampler', 'control depth', 'control_depth_Boreas')
#object = samp.SeaSampler('/home/marc/Projects/Mednet/tMednet/src/input_files/src/input_files/SeaSampler', 'metadata', 'metadata_seasampler_V5')

'''# Plot the radar with files given in different zones
object = samp.SeaSampler('/home/marc/Projects/Mednet/tMednet/src/input_files/SeaSampler Tossa', 'plots', 'plots')
object.add_data_list('/home/marc/Projects/Mednet/tMednet/src/input_files/SeaSampler Boreas')
object.add_data_list('/home/marc/Projects/Mednet/tMednet/src/input_files/SeaSampler Medes')
object.add_data_list('/home/marc/Projects/Mednet/tMednet/src/input_files/SeaSampler Ullastres')
object.radar_plot()'''

# Plot the radar with lotta files
lotsafiles = samp.SeaSampler('/home/marc/Projects/Mednet/tMednet/src/input_files/SeaSampler Corrected 2', 'radar_file_by_file', 'all dives')
lotsafiles.radar_plot_dives()


#object.dir_reader('no', '/home/marc/Projects/Mednet/tMednet/src/input_files/Copernicus September', object.open_netcdf)
#object.plot_surface_mean()
#object.create_map()

# Create map with temperature an dives
#lotsafiles.map_temperature('../medsea_omi_tempsal_extreme_var_temp_mean_and_anomaly.nc', '../SeaSampler_Dives.xlsx')