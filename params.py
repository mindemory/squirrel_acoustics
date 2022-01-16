import os
PROJECT_PATH = os.path.join('/Users/mrugankdake/Documents/Sciurid Lab/')

species_list = ['palmarum', 'pennanti', 'tristriatus', 'sublineatus']

accepted_columns = ['Selection',	'View',	'Channel',	'Begin Time (s)',	'End Time (s)',	'Low Freq (Hz)',
                    'High Freq (Hz)',	'Begin File',	'Begin Path', 'End File',	'End Path',	'File Offset (s)',
                    'Agg Entropy (bits)',	'Avg Amp (U)',	'Avg Entropy (bits)',	'Avg Power Density (dB FS)',
                    'BW 50% (Hz)',	'BW 90% (Hz)',	'Begin Clock Time',	'Begin Date',	'Begin Date Time',
                    'Beg File Samp (samples)',	'Begin Hour',	'Begin Sample (samples)',	'Center Freq (Hz)',
                    'Center Time (s)',	'Center Time Rel.',	'Delta Freq (Hz)',	'Delta Power (dB FS)',
                    'Delta Time (s)',	'Dur 50% (s)',	'Dur 90% (s)',	'End Clock Time',	'End Date',
                    'End File Samp (samples)',	'End Sample (samples)',	'Energy',	'F-RMS Amp (U)', 'Freq 25% (Hz)',
                    'Freq 5% (Hz)',	'Freq 75% (Hz)',	'Freq 95% (Hz)',	'Freq Contour  5% (Hz)',	'Freq Contour 25% (Hz)',
                    'Freq Contour 50% (Hz)',	'Freq Contour 75% (Hz)',	'Freq Contour 95% (Hz)',	'Inband Power (dB FS)',
                    'Length (frames)',	'Leq (dB FS)',	'Max Amp (U)',	'Max Bearing (deg)',	'Max Entropy (bits)',
                    'Max Freq (Hz)',	'Max Time (s)',	'Min Amp (U)',	'Min Entropy (bits)',	'Min Time (s)',	'Peak Amp (U)',
                    'Peak Corr (U)',	'Peak Freq (Hz)',	'Peak Freq Contour (Hz)',	'PFC Avg Slope (Hz/ms)',	'PFC Max Freq (Hz)',
                    'PFC Max Slope (Hz/ms)',	'PFC Min Freq (Hz)',	'PFC Min Slope (Hz/ms)',	'PFC Num Inf Pts',
                    'PFC Slope (Hz/ms)',	'Peak Lag (s)',	'Peak Power Density (dB FS)',	'Peak Time (s)',
                    'Peak Time Relative',	'RMS Amp (U)',	'SEL (dB FS)',	'SNR NIST Quick (dB FS)',	'Sample Length (samples)',
                    'Time 25% (s)',	'Time 25% Rel.',	'Time 5% (s)',	'Time 5% Rel.',	'Time 75% (s)',	'Time 75% Rel.',
                    'Time 95% (s)',	'Time 95% Rel.', 'Note', 'Quality',	'Bout',	'Sub-bout']

accepted_elements_note = ['LD', 'RD', 'IU', 'IU-RD', 'IU-LD', 'S', 'IS', 'NS', 'P', 'W', 'IU-RD-RE', 'IU-RD-LE', 'IU-LD-RE', 'RD-RE', 'LD-RE', 'IU-RE', 'T',
                     'lLD', 'lRD', 'lIU', 'lIU-RD', 'lIU-LD', 'lS', 'lIS', 'lP', 'lIU-LD-RE', 'lX3',
                     'llLD', 'llRD', 'llIU', 'llIU-RD', 'llIU-LD', 'llS', 'llIS', 'llP',  'lIU-RD-RE']

z_score = 1.96

bout_difference_dict = {'palmarum': 0.83, 'pennanti': 1.39, 'tristriatus': 0.71, 'sublineatus': 2.1}
sample_threshold_dict = {'F. palmarum': 33, 'F. pennanti': 33, 'F. tristriatus': 70, 'F. sublineatus': 40}

sub_bout_difference_dict = {'palmarum': 0.15, 'pennanti': 0.1, 'tristriatus': 0.15, 'sublineatus' : 0.1}
sub_bout_difference_dict_F = {'F. palmarum': 0.15, 'F. pennanti': 0.1, 'F. tristriatus': 0.15, 'F. sublineatus' : 0.1}
bins_dict = {'palmarum': 38000, 'pennanti': 15000, 'tristriatus': 6000, 'sublineatus': 4000}

columns = ['File_name', 'Species', 'Low Freq (Hz)', 'High Freq (Hz)', 'Delta Freq (Hz)','Delta Time (s)',
                    'Inter_note_difference (s)', 'Location', 'Latitude', 'Longitude', 'Note', 'Quality', 'Bout', 'Sub-bout']
acoustic_features = ['PFC Max Freq (Hz)',
        'PFC Min Freq (Hz)','BW 90% (Hz)', 'Agg Entropy (bits)','PFC Max Slope (Hz/ms)', 'PFC Avg Slope (Hz/ms)',
       'PFC Min Slope (Hz/ms)', 'Center Time Rel.', 'Dur 90% (s)', 'PFC Num Inf Pts']
#acoustic_features = ['Low Freq (Hz)', 'High Freq (Hz)', 'Delta Freq (Hz)','Delta Time (s)', 'Inter_note_difference (s)']
numerical_columns = ['Low Freq (Hz)', 'High Freq (Hz)', 'Delta Freq (Hz)','Delta Time (s)', 'Inter_note_difference (s)']
numerical_columns_with_fname = ['File_name', 'Low Freq (Hz)', 'High Freq (Hz)', 'Delta Freq (Hz)','Delta Time (s)', 'Inter_note_difference (s)']
#numerical_columns_with_fname = ['File_name', 'PFC Max Freq (Hz)', 'PFC Min Freq (Hz)', 'BW 90% (Hz)', 'Agg Entropy (bits)']

cat_columns = ['File_name', 'Species', 'Location', 'Latitude', 'Longitude']
