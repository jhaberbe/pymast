import numpy as np
import pandas as pd
import scipy.special

from typing import Union

from .pymast import HurdleLogNormal

# ideally, we want to be able to convert this to some analogous version of the output found in MAST. Maybe make it neater?

def onehot_encode(df: pd.DataFrame, columns: Union[list[str], None] = None):
    """_summary_

    Args:
        df (pd.DataFrame): pd.DataFrame of categorical variables

    Returns:
        _type_: _description_
    """
    if columns == None:
        # we assume that all the columns are to be converted

        # NOTE: we can also have the default be to select all columns with 
        # string or categorical datatypes, i'm not going to do that because
        # of the weirdness of object feeling a default dtypes in some cases.
        columns = df.columns

    for column in df.columns:
        onehot_df = pd.get_dummies(df[column], prefix=column)
        df = df.drop(column, axis=1)
        df = pd.concat([df, onehot_df], axis=1)
    return df

def grab_model_coefficients(model: HurdleLogNormal) -> pd.DataFrame:
    # Model coefficients.
    if model.is_fitted_:
        table = pd.DataFrame({
            "log_coef": model.logistic.coef_[0],
            "reg_coef": model.linear.named_steps['regressor'].coef_
        }, index=model._features)
    
        return model
    
    else:
        raise ValueError("Model has not been fitted yet.")

def compute_logfold_changes(model: HurdleLogNormal, original_log_base: float = np.exp(1)):
    """Computes the log-fold change for a given feature in a HurdleLogNormal model.

    Args:
        model (HurdleLogNormal): Trained Model
        feature (str): _description_
        original_log_base (float, optional): _description_. Defaults to np.exp(1).

    Returns:
        pd.Series: series of log2fold changes for each covariate given a feature.
    """
    
    assert model.is_fitted_, "Model has not been fitted yet."

    coefficients = grab_model_coefficients(model)

    intercept_value = scipy.special.expit(coefficients.loc["Intercept", "log_coef"]) * \
        scipy.special.expit(coefficients.loc["Intercept", "log_coef"])

    covariate_value = scipy.special.expit(coefficients.loc["Intercept", "log_coef"] + coefficients["log_coef"]) * \
        scipy.special.expit(coefficients.loc["Intercept", "reg_coef"] + coefficients["reg_coef"])

    # change of base function
    correction_factor = (np.log(original_log_base)/np.log(2))

    return pd.Series(data=(covariate_value / intercept_value) / correction_factor, index=model._features)

