import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

def R_call(R_file):
    utils = importr('utils')
    script = open(R_file)
    script = script.read()
    robjects.r(script)

