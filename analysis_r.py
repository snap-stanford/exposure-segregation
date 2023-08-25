"""
This file contains R code for analysis.
Currently just fitting the mixed model with R.

A separate file is used for R code so that R does not need to be installed
to use most of the code in analysis.py
"""
import pandas as pd
from rpy2.rinterface import NULLType
from rpy2.rinterface_lib.embedded import RRuntimeError
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from rpy2.robjects import Formula, pandas2ri

def fit_mixed_model_for_path_crossing_segregation(df):
    """
    Fits mixed model using R.
    df should be a dataframe with columns 'person', 'ses' and 'other_person_ses'
    where each row represents an SES pair for one path crossing
    and 'person' is a unique identifier for the person corresponding to 'ses'

    The code here is based on these references (but converted to use rpy2):
    - http://bodowinter.com/tutorial/bw_LME_tutorial.pdf

    Note: Although lme4 is used, the following nlme model appears to be the same:
      lme(other_person_ses ~ ses, random= ~ 1 | person, data=r_df)
    """
    base = importr('base')
    stats = importr('stats')
    lme4 = importr('lme4')
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df)
    ro.globalenv['r_df'] = r_df
    r_code = '''
        withWarnings <- function(expr) {
            myWarnings <- NULL
            wHandler <- function(w) {
                myWarnings <<- c(myWarnings, list(w))
                invokeRestart("muffleWarning")
            }
            val <- withCallingHandlers(expr, warning = wHandler)
            list(value = val, warnings = myWarnings)
        }
        withWarnings(lmer(other_person_ses ~ ses + (1|person), r_df))
    '''
    try:
        result = ro.r(r_code)
        lme_object = result.rx2('value')
        raw_warnings = result.rx2('warnings')
        converged = True
        warnings = []
        if not isinstance(raw_warnings, NULLType):
            converged = False
            warnings = [warning.rx2('message')[0] for warning in raw_warnings]
        print(base.summary(lme_object))
        print('Log likelihood:', stats.logLik(lme_object)[0])
        fe_params = lme4.getME(lme_object, 'fixef')
        ro.globalenv['lme_object'] = lme_object
        var_corr = ro.r('as.data.frame(VarCorr(lme_object))')
        return {
            'a': fe_params.rx2('ses')[0],
            'b': fe_params.rx2('(Intercept)')[0],
            'random_effects_covariance': var_corr.rx2(1, 'vcov')[0],
            'converged': converged,
            'warnings': warnings,
        }
    except RRuntimeError as e:
        return {
            'error': e.args,
        }
