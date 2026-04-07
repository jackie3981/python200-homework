# warmup exercises
# %%
# FOR INTERACTIVE WINDOWS ONLY: Uncomment the following lines to enable autoreload of modules
#%load_ext autoreload
#%autoreload 2
# *******************************************************************
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import warmup.pandas_review as pr
import warmup.numpy_review as nr
import warmup.matplotlib_review as mr
import warmup.descriptive_stats_review as dsr
import warmup.hypothesis_review as hr
import warmup.correlation_review as cr
import warmup.pipeline as pln
import prefect_warmup as pf
# *******************************************************************


#  %%
print("Pandas Review:")
#pr.pandas_review()

# %%
print("\nNumPy Review:")
#nr.numpy_review()

# %%
print("\nMatplotlib Review:")
#mr.matplotlib_review()

# %%
print("\nDescriptive Statistics Review:")
#dsr.descriptive_stats_review()

# %%
print("\nHypothesis Testing Review:")
#hr.hypothesis_review()

# %%
print("\nCorrelation Review:")
#cr.correlation_review()

# %%
print("\nPipeline Review:")
pln.pipeline_review()

# %%
print("\nPrefect Pipeline Review:")
pf.prefect_pipeline_review()