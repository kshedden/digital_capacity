import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from read import dm, dmp

cm = plt.get_cmap("tab10")
colors = {"i": cm(0), "k": cm(1/10), "a": cm(2/10)}
tm = {"mich": "Michigan", "nonmich": "Non-Michigan", "detroit": "Detroit",
      "parkside": "Parkside", "allvalid": "All valid responses"}

pdf = PdfPages("factor.pdf")
out = open("factor.txt", "w")

def loglike(S, n):
    p = S.shape[0]
    c = (n*p/2) * np.log(2*np.pi)
    _, ldet = np.linalg.slogdet(S)
    return -c - n*ldet/2 - n*p/2


for (ky, df) in dm.items():

    print(ky)

    # All dataframes should have the same columns
    if ky == "mich":
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

    cn = df.columns
    cn = [x.replace("know_", "k_") for x in cn]
    cn = [x.replace("internet_", "i_") for x in cn]
    df.columns = cn

    # Conside factor solutions of dimension 1-3
    frl, aic, bic, scores = [], [], [], []
    for d in range(1, 10):
        X = np.asarray(df)
        Xc = X - X.mean(0)
        fa = sm.Factor(X, method="ml", n_factor=d)
        fx = fa.fit(maxiter=2000, opt_method='l-bfgs-b')
        #print("|grad| = ", np.linalg.norm(fx.mle_retvals.jac))
        n = df.shape[0]
        p = fx.model.k_endog
        llf = -fx.mle_retvals.fun * p * n
        degf = d*p - p*(p-1)/2
        aic.append(-2*llf + 2*degf)
        bic.append(-2*llf + np.log(n)*degf)
        fr1 = pd.DataFrame(fx.loadings, index=df.columns)
        frl.append(fr1)
        scores.append(np.dot(Xc, fx.loadings))
    aic = np.asarray(aic)
    aic -= aic.min()
    bic = np.asarray(bic)
    bic -= bic.min()
    print("AIC: ", np.argmin(aic) + 1)
    print("BIC: ", np.argmin(bic) + 1)

    # Use the dimension 2 solution
    fr = frl[1]
    scr = scores[1]

    # Flip for visualization
    for j in 0,1:
        if (fr.iloc[:, j] < 0).mean() > 0.5:
            fr.iloc[:, j] *= -1

    fr["0_r2"] = 0
    fr["1_r2"] = 0
    for i in range(2):
        fr.iloc[:, 2+i] = [np.corrcoef(scr[:, i], X[:, j])[0, 1] for j in range(X.shape[1])]

    out.write("==== %s (n=%d) ====\n" % (tm[ky], df.shape[0]))
    out.write(fr.to_string())
    out.write("\n\n")

    plt.clf()
    plt.grid(True)
    plt.title("%s (n=%d)" % (tm[ky], df.shape[0]))
    for i in range(fr.shape[0]):
        k = cn[i]
        plt.plot([0, fr.iloc[i, 0]], [0, fr.iloc[i, 1]], "-", color=colors[k[0]])
        plt.text(fr.iloc[i, 0], fr.iloc[i, 1], str(i + 1))
    plt.xlabel("Component 1", size=15)
    plt.ylabel("Component 2", size=15)
    pdf.savefig()

    # Correlations between passive variables and factor scores
    fr = []
    for k in range(2):
        for j in range(1, dp.shape[1]):
            u = dp.iloc[:, j].unique()
            u = [x for x in u if not pd.isnull(x)]
            u.sort()
            for i in range(len(u)):
                x = scr[:, k]
                y = (dp.iloc[:, j] == u[i]).values
                ii = pd.notnull(x) & pd.notnull(y)
                x = x[ii]
                y = y[ii]
                row = [dp.columns[j], u[i], k + 1, len(ii), np.corrcoef(x, y)[0, 1]]
                fr.append(row)
    dr = pd.DataFrame(fr, columns=["Variable", "Value", "Component", "N", "Correlation"])
    dr["Z"] = np.sqrt(dr["N"]) * dr["Correlation"]
    out.write(dr.to_string(index=None))
    out.write("\n\n")

pdf.close()
out.close()
