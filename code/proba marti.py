import seasampler as samp

#object = samp.SeaSampler('../src/input_files/Boreas', 'control depth', 'control_depth_Boreas')
#object = samp.SeaSampler('../src/input_files/SeaSampler', 'metadata', 'metadata_seasampler_V4')
object = samp.SeaSampler('/home/marc/Projects/Mednet/tMednet/src/input_files/SeaSampler Tossa', 'plots', 'plots')
#object.add_data_list('../src/input_files/SeaSampler Boreas')
#object.add_data_list('../src/input_files/SeaSampler Medes')
#object.add_data_list('../src/input_files/SeaSampler Ullastres')
#object.radar_plot()
#object.radar_plot()
object.dir_reader('no', '/home/marc/Projects/Mednet/tMednet/src/input_files/Copernicus September', object.open_netcdf)
#object.plot_surface_mean()
#object.create_map()
object.map_temperature('../medsea_omi_tempsal_extreme_var_temp_mean_and_anomaly.nc', '../SeaSampler_Dives.xlsx')