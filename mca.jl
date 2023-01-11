using MultivariateStats
using PyPlot
using Printf

rm("plots", recursive = true, force = true)
mkdir("plots")

out = open("mca.txt", "w")

include("read.jl")

cm = PyPlot.get_cmap("tab10")
colors = Dict(
    "i" => cm(0),
    "k" => cm(1 / 10),
    "a" => cm(2 / 10),
)

tm = Dict(
    "mich" => "Michigan",
    "nonmich" => "Non-Michigan",
    "detroit" => "Detroit",
    "parkside" => "Parkside",
    "all" => "All locations",
)

function make_plots(ifig)
    for (k, df) in dm
        #df = df[completecases(df), :]
        vn = names(df)
        vn = [replace(x, "know_" => "k_") for x in vn]
        vn = [replace(x, "internet_" => "i_") for x in vn]
        rename!(df, vn)

        #df = disallowmissing(df)
        for c in names(df)
            if eltype(df[:, c]) <: Float64
                df[!, c] = Int.(df[:, c])
            end
        end

        df, dp = if k == "all"
            dp = df[:, :reg]
            df = select(df, Not(:reg))
            df, dp
        else
            df, nothing
        end

        mca = fit(MCA, df; method="indicator", normalize = "principal", d = 2)
        write(out, @sprintf("==== %s ====\n\n", tm[k]))
        show(out, MIME("text/plain"), mca)
        cc = variable_coords(mca)
        cc = DataFrame(
            :Variable => cc.Variable,
            :Level => cc.Level,
            :X1 => cc.Coord[:, 1],
            :X2 => cc.Coord[:, 2],
        )

        PyPlot.clf()
        PyPlot.figure(figsize = (10, 8))
        PyPlot.grid(true)
        PyPlot.title("$(tm[k]) (n=$(size(df, 1)))")
        frst = Dict(
            "i" => true,
            "k" => true,
            "a" => true,
        )
        ii = 1
        for cx in groupby(cc, :Variable)
            ky = first(cx[:, :Variable])
            ky1 = string(ky[1])
            xx = cx[:, :X1]
            yy = cx[:, :X2]
            col = colors[ky1]
            if frst[ky1]
                PyPlot.plot(xx, yy, "-", color = col, alpha = 0.6, label = ky1)
                frst[ky1] = false
            else
                PyPlot.plot(xx, yy, "-", color = col, alpha = 0.6)
            end

            # Item numbers for the questionairre
            u = 0.2 * (2 * rand(2) .- 1)
            if ky1 in ["a", "i", "k"]
                PyPlot.text(xx[1] + u[1], yy[1] + u[2], string(ii) * "-", color = col)
                PyPlot.text(xx[end] + u[1], yy[end] + u[2], string(ii) * "+", color = col)
            end

            ii += 1
        end

        if k == "all"
            pc = quali_passive(mca, dp; normalize = "principal")
            xx = pc.Coord
            la = pc.Level
            for j in 1:4
                PyPlot.text(xx[j, 1], xx[j, 2], la[j])
            end
        end

        ha, lb = PyPlot.gca().get_legend_handles_labels()
        leg = PyPlot.figlegend(ha, lb, "center right")
        leg.draw_frame(false)
        PyPlot.xlabel("Component 1", size = 15)
        PyPlot.ylabel("Component 2", size = 15)
        PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
        ifig += 1
    end
    return ifig
end

ifig = 0
ifig = make_plots(ifig)

f = [@sprintf("plots/%03d.pdf", j) for j = 0:ifig-1]
c = `gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dAutoRotatePages=/None -sOutputFile=mca.pdf $f`
run(c)

close(out)

out = open("vars.csv", "w")
na = names(dm["mich"])
for i in eachindex(na)
    write(out, @sprintf("%d,%s\n", i, na[i]))
end
close(out)
