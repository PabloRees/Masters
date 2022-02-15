import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

utils = importr('utils')

def R_call(R_file):
    script = open(R_file)
    script = script.read()
    robjects.r(script)

R_call('GSPC2_logdiff.R')