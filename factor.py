import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from scipy.stats.distributions import norm
from read import dm, dmp

cm = plt.get_cmap("tab10")
colors = {"i": cm(0), "k": cm(1/10), "a": cm(2/10)}
tm = {"mich": "Michigan", "nonmich": "Non-Michigan", "detroit": "Detroit",
      "parkside": "Parkside", "allvalid": "All valid responses"}

def train(df):

    X = np.asarray(df)
    Xc = X - X.mean(0)

    # Consider factor solutions of dimension 1-3
    aic, bic, loadings, scores = [], [], [], []
    for d in range(1, 10):
        fa = sm.Factor(X, method="ml", n_factor=d)
        fx = fa.fit(maxiter=2000, opt_method='l-bfgs-b')
        #print("|grad| = ", np.linalg.norm(fx.mle_retvals.jac))
        n = df.shape[0]
        p = fx.model.k_endog
        llf = -fx.mle_retvals.fun * p * n
        degf = d*p - p*(p-1)/2
        aic.append(-2*llf + 2*degf)
        bic.append(-2*llf + np.log(n)*degf)
        loads = pd.DataFrame(fx.loadings, index=df.columns)
        loadings.append(loads)
        scores.append(np.dot(Xc, fx.loadings))
    aic = np.asarray(aic)
    aic -= aic.min()
    bic = np.asarray(bic)
    bic -= bic.min()
    print("AIC: ", np.argmin(aic) + 1)
    print("BIC: ", np.argmin(bic) + 1)

    # Use the dimension 2 solution
    load = loadings[1]
    load.columns = ["1_load", "2_load"]
    scr = scores[1]

    # Flip for visualization
    if (load.iloc[:, 0] < 0).mean() > 0.5:
        load.iloc[:, 0] *= -1
        scr[:, 0] *= -1
    if load.iloc[14, 1] < 0:
        load.iloc[:, 1] *= -1
        scr[:, 1] *= -1

    return load, scr

def loglike(S, n):
    p = S.shape[0]
    c = (n*p/2) * np.log(2*np.pi)
    _, ldet = np.linalg.slogdet(S)
    return -c - n*ldet/2 - n*p/2


def passive_cor(dp, scr, out):

    # Correlations between passive variables and factor scores
    fr = []
    for k in range(2):
        for j in range(1, dp.shape[1]):

            x = scr[:, k]
            y = dp.iloc[:, j].values
            ii = pd.notnull(x) & pd.notnull(y)
            x = x[ii]
            y = y[ii]

            # Quantitative passive variables
            if dp.columns[j] in ["age", "hhs"]:
                n = sum(ii)
                rr = np.corrcoef(x, y)[0, 1]
                zz = 0.5*np.sqrt(n)*np.log((1+rr)/(1-rr))
                p = 2*norm.cdf(-np.abs(zz))
                row = [dp.columns[j], "", k + 1, n, "", rr, zz, p]
                fr.append(row)
            elif dp.columns[j] in ["money", "education"]:
                rr = spearmanr(x, y)
                zz = np.sign(rr.correlation) * np.abs(norm.ppf(rr.pvalue/2))
                row = [dp.columns[j], "", k + 1, len(ii), "", rr.correlation, zz, rr.pvalue]
                fr.append(row)
            elif dp.columns[j] in ["work"]:
                rr = kendalltau(x, y)
                zz = np.sign(rr.correlation) * np.abs(norm.ppf(rr.pvalue/2))
                row = [dp.columns[j], "", k + 1, len(ii), "", rr.correlation, zz, rr.pvalue]
                fr.append(row)
            else:
                # Qualitative passive variables
                u = dp.iloc[:, j].unique()
                u = [x for x in u if not pd.isnull(x)]
                u.sort()
                for i in range(len(u)):
                    x = scr[:, k]
                    y = (dp.iloc[:, j] == u[i]).values
                    ii = pd.notnull(x) & pd.notnull(y)
                    x = x[ii]
                    y = y[ii]
                    if sum(y) < 10:
                        continue
                    n = sum(ii)
                    rr = np.corrcoef(x, y)[0, 1]
                    zz = 0.5*np.sqrt(n)*np.log((1+rr)/(1-rr))
                    p = 2*norm.cdf(-np.abs(zz))
                    row = [dp.columns[j], u[i], k + 1, n, sum(y), rr, zz, p]
                    fr.append(row)
    dr = pd.DataFrame(fr, columns=["Variable", "Value", "Component", "N", "N_value", "Correlation", "ZScore", "Pvalue"])
    out.write(dr.to_string(index=None))
    out.write("\n\n")

def regression(dp, scr, out):
    # Regression analysis of factor scores
    dq = dp.copy()
    dq["score1"] = scr[:, 0]
    dq["score2"] = scr[:, 1]
    for j in 1, 2:
        m0 = sm.OLS.from_formula("score%d ~ age + race + sex + hhs + money + education" % j, data=dq)
        r0 = m0.fit()
        out.write(r0.summary().as_text())
        out.write("\n\n")

def main(kyu, mode):

    modes = "_allvalidfactors" if mode == "allvalid" else ""
    pdf = PdfPages("factor%s.pdf" % modes)
    out = open("factor%s.txt" % modes, "w")

    for ky in kyu:

        print(ky)
        df = dm[ky]

        # All dataframes should have the same columns
        if ky == "allvalid":
            x = df.columns[1:]
            io = open("vars.csv", "w")
            for i in range(len(x)):
                io.write("%d,%s\n" % (i+1, x[i]))
            io.close()

        df = df.drop("Response_Id", axis=1)
        df = df.iloc[:, 0:29]
        ii = pd.notnull(df).all(1)
        df = df.loc[ii, :]
        dp = dmp[ky].loc[ii, :]
        X = np.asarray(df)
        Xc = X - X.mean(0)

        cn = df.columns
        cn = [x.replace("know_", "k_") for x in cn]
        cn = [x.replace("internet_", "i_") for x in cn]
        df.columns = cn

        if ky == "allvalid" or mode != "allvalid":
            load, scr = train(df)
        else:
            scr = np.dot(Xc, load)

        for i in range(2):
            load["%d_r" % (i + 1)] = [np.corrcoef(scr[:, i], X[:, j])[0, 1] for j in range(X.shape[1])]
            load["%d_r2" % (i + 1)] = load["%d_r" % (i + 1)]**2

        out.write("==== %s (n=%d) ====\n" % (tm[ky], df.shape[0]))
        out.write("\nCorrelations and squared correlations between each item and each factor:\n")
        out.write(load.to_string())
        out.write("\n\n")

        plt.clf()
        plt.grid(True)
        plt.title("%s (n=%d)" % (tm[ky], df.shape[0]))
        for i in range(load.shape[0]):
            k = cn[i]
            plt.plot([0, load.iloc[i, 0]], [0, load.iloc[i, 1]], "-", color=colors[k[0]])
            plt.text(load.iloc[i, 0], load.iloc[i, 1], str(i + 1))
        plt.xlabel("Component 1", size=15)
        plt.ylabel("Component 2", size=15)
        pdf.savefig()

        passive_cor(dp, scr, out)
        regression(dp, scr, out)

    pdf.close()
    out.close()

# If mode == allvalid then use the allvalid data to estimate the factors
# for all cohorts, if mode == each then use each cohort to estimate
# its own factors
mode = "allvalid"

kyu = ["allvalid", "parkside"]

for mode in "allvalid", "":
    main(kyu, mode)
