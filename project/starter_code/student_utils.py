import pandas as pd
import numpy as np
import os
import tensorflow as tf
import functools

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
     '''  
    ndc_df["Non-proprietary Name"]= ndc_df["Non-proprietary Name"].str.replace("Hcl", "Hydrochloride")
    ndc_df["Non-proprietary Name"]= ndc_df["Non-proprietary Name"].str.replace(" And ", "-") 
    
    ndc_df["Non-proprietary Name"]= (ndc_df["Non-proprietary Name"].str.strip()).str.upper()
    
#     ndc_df["Dosage Form"]= ndc_df["Dosage Form"].str.replace("Tablet, Film Coated", "TABLET")
#     ndc_df["Dosage Form"]= ndc_df["Dosage Form"].str.replace("Tablet, Coated", "TABLET")
#     ndc_df["Dosage Form"]= ndc_df["Dosage Form"].str.replace("Tablet, Film Coated, Extended Release", "Tablet Extended Release")
#     ndc_df["Dosage Form"]= ndc_df["Dosage Form"].str.replace("Tablet, Extended Release", "Tablet Extended Release")
#     ndc_df["Dosage Form"]= ndc_df["Dosage Form"].str.replace("For Suspension, Extended Release", "For Suspension Extended Release")
#     ndc_df["Dosage Form"]= ndc_df["Dosage Form"].str.replace("Powder, Metered", "Powder Metered")
#     ndc_df["Dosage Form"]= (ndc_df["Dosage Form"].str.strip()).str.upper()
#     ndc_df["generic_drug_name"]= ndc_df["Non-proprietary Name"]+"_"+ndc_df["Dosage Form"]
    ndc_df["generic_drug_name"]= ndc_df["Non-proprietary Name"]
    df_reduce_dimension = pd.merge(df, ndc_df, on=['ndc_code'], how='inner')
    df_reduce_dimension['LABEL'] = 0
    reduce_dim_df= df_reduce_dimension.drop(columns=['Proprietary Name', 'Non-proprietary Name', 'Dosage Form', 'Route Name', 'Company Name', 'Product Type'])
    
    return reduce_dim_df 


#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    
    first_encounter_df = df.sort_values('encounter_id').groupby('patient_nbr').first()
    first_encounter_df = first_encounter_df.reset_index()
    return first_encounter_df
 


#Question 6
def patient_dataset_splitter(df, key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    df = df.iloc[np.random.permutation(len(df))]
    unique_values = df[key].unique()
    total_values = len(unique_values)
    train_size = round(total_values * (1 - 0.4 ))
    train = df[df[key].isin(unique_values[:train_size])].reset_index(drop=True)
    left_size = len(unique_values[train_size:])
    validation_size = round(left_size*0.5)

    validation = df[df[key].isin(unique_values[train_size:train_size+validation_size])].reset_index(drop=True) 
    test = df[df[key].isin(unique_values[validation_size+train_size:])].reset_index(drop=True) 
    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key=c, vocabulary_file = vocab_file_path, num_oov_buckets=1)
        one_hot_origin_feature = tf.feature_column.indicator_column(tf_categorical_feature_column)    
        output_tf_list.append(one_hot_origin_feature)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature= tf.feature_column.numeric_column(
    key=col, default_value = default_value, normalizer_fn=normalizer, dtype=tf.float64)
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    
    def convert_to_binary(df, pred_field, actual_field):
    df['score'] = df[pred_field].apply(lambda x: 1 if x>=25 else 0 )
    df['label_value'] = df[actual_field].apply(lambda x: 1 if x>=25 else 0)
    
    return df
    binary_df = convert_to_binary(model_output_df, 'pred', 'actual_value')
binary_df.head()
    
    '''
    return student_binary_prediction
