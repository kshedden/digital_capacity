import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from read import dm

cm = plt.get_cmap("tab10")
colors = {"i": cm(0), "k": cm(1/10), "a": cm(2/10)}
tm = {"mich": "Michigan", "nonmich": "Non-Michigan", "detroit": "Detroit", "parkside": "Parkside", "all": "All locations"}

pdf = PdfPages("factor.pdf")
out = open("factor.txt", "w")

for (ky, df) in dm.items():
    print(ky)
    df = df.drop("Response_Id", axis=1)
    df = df.dropna()
    df = df.iloc[:, 0:29]
    cn = df.columns
    cn = [x.replace("know_", "k_") for x in cn]
    cn = [x.replace("internet_", "i_") for x in cn]
    df.columns = cn

    # Conside factor solutions of dimension 1-3
    frl = []
    for d in 1, 2, 3:
        fa = sm.Factor(np.asarray(df), method="ml", n_factor=d)
        fx = fa.fit()
        n = df.shape[0]
        print("loglike=", -fx.mle_retvals.fun * fx.model.k_endog * n, " ", fx.mle_retvals.x.size)
        fr1 = pd.DataFrame(fx.loadings, index=df.columns)
        frl.append(fr1)

    # Use the dimension 2 solution
    fr = frl[1]

    # Flip for visualization
    for j in 0,1:
        if (fr.iloc[:, j] < 0).mean() > 0.5:
            fr.iloc[:, j] *= -1

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

pdf.close()
out.close()
