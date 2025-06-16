from math import floor
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import fooof
import pickle


with open("G:\\Drive condivisi\\AnalisiSegnaliEeg\\HD EEG\\Crisi_HDEEG\\ICA_pruned_reduced\\plot_indeces_organization", 'rb') as file:
    new_indices = pickle.load(file)


def baseline_correction_separate(power_data):
    """
    Applies baseline correction separately for each subject and each ROI.

    Parameters:
    - power_data: dict {subject_name: np.array of shape (68, 120, 2346)} 
      where each key corresponds to a subject, and values are the power spectrum.

    Returns:
    - corrected_data: dict {subject_name: np.array of shape (68, 120, 2346)}
      with baseline correction applied.
    """
    corrected_data = {}  # Dictionary to store results

    # Iterate over subjects
    for idx, (subj, values) in enumerate(power_data.items()):  # 29 subjects
        corrected_values = np.zeros_like(values)  # Initialize array for subject
        
        for roi in range(values.shape[0]):  # 68 ROIs
            baseline_activity = values[roi, 0, :]  # Baseline (shape: 2346)

            # Avoid division by zero
            baseline_activity = np.where(baseline_activity == 0, 1e-10, baseline_activity)

            # Apply correction across timepoints
            corrected_values[roi, :, :] = 10 * np.log10(values[roi, :, :] / baseline_activity)

        corrected_data[subj] = corrected_values  # Store corrected data for each subject

    return corrected_data

############################################################################################
def reorganize_rois(original_data, new_indices=new_indices):
  '''
  A function that reorder the atlas parcels for plot purposes

  Arguments
  ---------
    original_data: a list.
      the original wrong ordered data, output of reorder_hemispheres function
    new_indices: a list.
      a list of the indeces in the new order
  Returns
  -------
    x: an array.
      the reordered data used for the cortical plot
  '''
  x = [original_data[i] for i in new_indices]
  return(np.array(x))
############################################################################################

############################################################################################
def reorder_hemispheres(original_data, first_left = True):
  '''
  A function that reorder the atlas parcels for plot purposes.

  Arguments
  ---------
    original_data: a list.
      the original wrong ordered data, output of reorder_hemispheres function
    first_left: a boolean.
      True if we want all left hemispheres parcels before right hemisphere ones,
      False otherwise
  Returns
  -------
    ordered_data: a list.
      the reordered data used for the cortical plot. This will become the input for
      the reorganize_roi_gradient function
  '''
  left_h = []
  right_h = []
  for idx, val in enumerate(original_data):
    idx +=1
    if idx%2==0:
      right_h.append(val)
    else:
      left_h.append(val)
  if first_left:
    ordered_data = np.concatenate((left_h, right_h))
  else:
    ordered_data = np.concatenate((right_h, left_h))
  return ordered_data.tolist()
############################################################################################

############################################################################################
def mat_struct_to_dict(mat_obj):
    """ Recursively convert MATLAB structs to Python dictionaries """
    if isinstance(mat_obj, scipy.io.matlab.mio5_params.mat_struct):
        return {field: mat_struct_to_dict(getattr(mat_obj, field)) for field in mat_obj._fieldnames}
    elif isinstance(mat_obj, list):
        return [mat_struct_to_dict(item) for item in mat_obj]
    else:
        return mat_obj
############################################################################################


############################################################################################
def load_aperiodic_from_mat_files(parent_directory, aperiodic_component = 'exponent'):
    """
    Loads and extracts FOOOF aperiodic component data from all `.mat` files in the specified parent directory.
    This function iterates over all `.mat` files in the given directory, processes them by 
    and stores the results in a dictionary.

    Parameters:
    -----------
    parent_directory : str
        Path to the directory containing `.mat` files.
    aperiodic_component = str
        choose wich aperioding component to extract between 'exponent' and 'offset'

    Returns:
    --------
    data_dict : dict
        A dictionary where:
            - Keys are the subject names (extracted from file names, removing '_Sprint' if present).
            - Values are the extracted FOOOF exponent data from the `.mat` file.

    Notes:
    ------
    - Only files ending with `.mat` are considered.
    - The function assumes that the key in the loaded `.mat` structure matches the filename (without `.mat`).
    - Requires `scipy.io.loadmat` to load `.mat` files and `natsort.natsorted` for natural sorting.
    """
    # Initialize an empty dictionary to store results
    data_dict = {}

    # Iterate over each file in the parent directory, sorted naturally
    for file_name in natsorted(os.listdir(parent_directory)):  
        if file_name.endswith('.mat'):  # Consider only .mat files
            file_path = os.path.join(parent_directory, file_name)
            key_name = file_name.replace('.mat', '')
            dict_key = key_name.replace('_Sprint', '')

            print(f"Loading subject {dict_key} and extracting FOOOF exponent data...")

            # Load .mat file
            mat_data = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)

            # Convert MATLAB structure to dictionary
            sprint_dict = mat_struct_to_dict(mat_data[key_name])

            # Extract and store exponent data
            data_dict[dict_key] = sprint_dict['Options']['SPRiNT']['topography'][aperiodic_component]

            # Free memory
            del sprint_dict  

    return data_dict
############################################################################################

############################################################################################
def compare_slopes(y1, y2):
    """
    Statistically compares the slopes of two time series using a t-test.
    
    Parameters:
        y1 (array-like): First time series (e.g., selected indices).
        y2 (array-like): Second time series (e.g., non-selected indices).
        
    Returns:
        dict: A dictionary containing the slopes, standard errors, t-statistic, and p-value.
    """
    # Create x values (time)
    x1 = np.arange(len(y1))
    x2 = np.arange(len(y2))
    
    # Fit OLS models
    model1 = sm.OLS(y1, sm.add_constant(x1)).fit()
    model2 = sm.OLS(y2, sm.add_constant(x2)).fit()
    
    # Extract slopes and standard errors
    beta1, se1 = model1.params[1], model1.bse[1]
    beta2, se2 = model2.params[1], model2.bse[1]
    
    # Compute t-statistic
    t_stat = (beta1 - beta2) / np.sqrt(se1**2 + se2**2)
    
    # Conservative degrees of freedom
    df = min(len(y1) - 2, len(y2) - 2)
    
    # Compute two-sided p-value
    p_value = 2 * stats.t.sf(np.abs(t_stat), df)
    
    return {
        "slope1": beta1,
        "slope2": beta2,
        "se1": se1,
        "se2": se2,
        "t_stat": t_stat,
        "p_value": p_value
    }
############################################################################################

############################################################################################
def compute_avg_std_se(nested_list):
    flat = [nested_list[i][j] for i in range(len(nested_list)) for j in range(len(nested_list[i]))]
    data = np.stack(flat)
    avg = np.mean(data, axis=0)
    std = np.std(data, axis=0, ddof=1)
    se = std / np.sqrt(data.shape[0])
    return avg, std, se
############################################################################################





