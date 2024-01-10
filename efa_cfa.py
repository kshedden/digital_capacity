import statsmodels.api as sm
from statsmodels.multivariate.factor import CFABuilder
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats.distributions import norm
from utils import *

pdf = PdfPages("factor.pdf")

# Load JFS
from jfs_read import dm as jfs, jfs_demog as jfs_c

# Load FOP (Parkside)
from fop_read import dm, dmp
fop = dm["parkside"]
fop_c = dmp["parkside"]

# Drop people with missing factor items.
ii = pd.notnull(fop).all(1)
fop = fop.loc[ii, :]
fop_c = fop_c.loc[ii, :]
ii = pd.notnull(jfs).all(1)
jfs = jfs.loc[ii, :]
jfs_c = jfs_c.loc[ii, :]

# These items are included in the EFA but not the CFA.
drop = ["a_spreadsheet", "k_place", "k_prob", "k_daily", "i_reliable", "i_address", "i_outside", "i_place"]

def reorder(df):
    a = [x for x in df.columns if x.startswith("a_")]
    i = [x for x in df.columns if x.startswith("i_")]
    k = [x for x in df.columns if x.startswith("k_")]
    v = ["Response_Id"] + a + i + k
    df = df[v]
    return df

jfs = reorder(jfs)
fop = reorder(fop)

def write_vars(da, na):
    with open("vars_%s.csv" % na, "w") as io:
        xx = da.columns[1:].tolist()
        for i in range(len(xx)):
            io.write("%d,%s\n" % (i+1, xx[i]))

def plot_compare_means(jfs, fop):

    # Items to analyze
    qs = jfs.columns[1:].tolist()

    fop_means = fop[qs].mean(0)
    jfs_means = jfs[qs].mean(0)
    jfs_means.name = "JFS"
    fop_means.name = "Parkside"
    cohort_means = pd.merge(fop_means, jfs_means, left_index=True, right_index=True)
    cohort_means.index.name = "item"
    cohort_means.to_csv("cohort_means.csv")

    # Make two scatterplots, one with item names and one with item numbers
    for vi in (0, 1):
        plt.clf()
        plt.grid(True)
        plt.plot(cohort_means.iloc[:, 0], cohort_means.iloc[:, 1], alpha=0)
        plt.gca().axline((1,1), slope=1)
        for i in range(cohort_means.shape[0]):
            na = cohort_means.index[i]
            c = colors[na[0]]
            if vi == 0:
                plt.text(cohort_means.iloc[i, 0], cohort_means.iloc[i, 1], na, color=c, ha="center", va="center")
            else:
                plt.text(cohort_means.iloc[i, 0], cohort_means.iloc[i, 1], "%.0f" % (i+1), color=c, ha="center", va="center")
        plt.xlabel("FOP mean", size=18)
        plt.ylabel("JFS mean", size=18)
        pdf.savefig()


def do_factor(dm, dc, dd, drop=None, do_cfa=False):
    dx = dm.drop("Response_Id", axis=1)
    ii = pd.notnull(dx).all(1)
    dx = dx.loc[ii, :]

    if drop is not None:
        dx = dx.drop(columns=drop)
        write_vars(dx, "cfa")
    else:
        write_vars(dx, "efa")

    cfamat = None
    if do_cfa:
        a_items = [x for x in dx.columns if x.startswith("a_")]
        i_items = [x for x in dx.columns if x.startswith("i_")]
        k_items = [x for x in dx.columns if x.startswith("k_")]
        cfamat = CFABuilder.no_crossload(dx, [a_items, k_items])

    fa, rslt = fit_fa(dx, dd, cfa=cfamat)
    load = fa.loadings
    scr = np.dot(dx, load)
    load, scr = flip(load, scr)
    d = load.shape[1]
    load = pd.DataFrame({"f%d" % (j+1): load[:, j] for j in range(d)}, index=dx.columns)
    load_se = rslt.load_stderr
    srmr, srmrv = rslt.srmr
    load["srmr"] = srmrv
    load["uniqueness"] = fa.uniqueness

    return dx, dc.loc[ii, :], load, scr, srmr

def do_factor_regress(scr, dc, out, va):
    dd = pd.DataFrame({"f1": scr[:, 0], "f2": scr[:, 1]})
    dd[va] = dc[va]
    fml = " + " + " + ".join(va)

    mm = sm.OLS.from_formula("f1 ~ " + fml, dd).fit()
    out.write("Factor 1:\n")
    out.write(mm.summary().as_text())
    out.write("\n\n")
    out.write("Factor 2:\n")
    mm = sm.OLS.from_formula("f2 ~ " + fml, dd).fit()
    out.write(mm.summary().as_text())
    out.write("\n\n")

def plot_factor_loadings(load, dx, na):
    plt.clf()
    plt.grid(True)
    plt.title("%s (n=%d)" % (na, dx.shape[0]))
    for i in range(load.shape[0]):
        k = dx.columns[i]
        plt.plot([0, load.iloc[i, 0]], [0, load.iloc[i, 1]], "-", color=colors[k[0]])
        plt.text(load.iloc[i, 0], load.iloc[i, 1], str(i + 1))
    plt.xlabel("Component 1", size=15)
    plt.ylabel("Component 2", size=15)
    pdf.savefig()

def save_loadings(out, na, load, srmr):
    out.write("=== %s ====\n" % na)
    out.write("Overall SRMR: %f\n\n" % srmr)
    out.write(load.to_string())
    out.write("\n\n")

plot_compare_means(jfs, fop)

dd = 2

# Do EFA and CFA for FOP
fop_efa, fop_c_efa, efa_load_fop, efa_scr_fop, efa_srmr_fop = do_factor(fop, fop_c, dd, do_cfa=False)
fop_cfa, fop_c_cfa, cfa_load_fop, cfa_scr_fop, cfa_srmr_fop = do_factor(fop, fop_c, dd, do_cfa=True, drop=drop)

# Do EFA and CFA for JFS
jfs_efa, jfs_c_efa, efa_load_jfs, efa_scr_jfs, efa_srmr_jfs = do_factor(jfs, jfs_c, dd, do_cfa=False)
jfs_cfa, jfs_c_cfa, cfa_load_jfs, cfa_scr_jfs, cfa_srmr_jfs = do_factor(jfs, jfs_c, dd, do_cfa=True, drop=drop)

plot_factor_loadings(efa_load_fop, fop_efa, "FOP EFA")
plot_factor_loadings(efa_load_jfs, jfs_efa, "JFS EFA")

plot_factor_loadings(cfa_load_fop, fop_cfa, "FOP CFA")
plot_factor_loadings(cfa_load_jfs, jfs_cfa, "JFS CFA")

with open("factor_regressions.txt", "w") as out:
    out.write("==== FOP EFA ====\n")
    do_factor_regress(efa_scr_fop, fop_c_efa, out, ["age", "sex"])
    do_factor_regress(efa_scr_fop, fop_c_efa, out, ["age", "sex", "hhs", "education", "work", "money"])
    out.write("==== FOP CFA ====\n")
    do_factor_regress(cfa_scr_fop, fop_c_cfa, out, ["age", "sex"])
    do_factor_regress(cfa_scr_fop, fop_c_cfa, out, ["age", "sex", "hhs", "education", "work", "money"])
    out.write("==== JFS EFA ====\n")
    do_factor_regress(efa_scr_jfs, jfs_c_efa, out, ["age", "sex", "UserLanguage"])
    do_factor_regress(efa_scr_jfs, jfs_c_efa, out, ["age", "sex", "UserLanguage", "hhs", "education"])
    out.write("==== JFS CFA ====\n")
    do_factor_regress(cfa_scr_jfs, jfs_c_cfa, out, ["age", "sex", "UserLanguage"])
    do_factor_regress(cfa_scr_jfs, jfs_c_cfa, out, ["age", "sex", "UserLanguage", "hhs", "education"])

with open("loadings.txt", "w") as out:
    save_loadings(out, "FOP EFA", efa_load_fop, efa_srmr_fop)
    save_loadings(out, "FOP CFA", cfa_load_fop, cfa_srmr_fop)
    save_loadings(out, "JFS EFA", efa_load_jfs, efa_srmr_jfs)
    save_loadings(out, "JFS CFA", cfa_load_jfs, cfa_srmr_jfs)

pdf.close()
