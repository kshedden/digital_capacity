using MultivariateStats
using PyPlot
using Printf
using Statistics

rm("plots", recursive = true, force = true)
mkdir("plots")

out = open("mca.txt", "w")

include("read.jl")

cm = PyPlot.get_cmap("tab10")
colors = Dict("i" => cm(0), "k" => cm(1 / 10), "a" => cm(2 / 10))

tm = Dict(
    "mich" => "Michigan",
    "nonmich" => "Non-Michigan",
    "detroit" => "Detroit",
    "parkside" => "Parkside",
    "allvalid" => "All valid responses",
)

function passive_summary(out, xc, dp)

    vnames = names(dp)
    write(out, "Correlations between passive variables and components:\n")

    dr = DataFrame(:Group=>String[], :Component=>Int[], :N=>Int[], :Corr=>Float64[], :Z=>Float64[])
    for j in 1:size(xc, 2)
        for k in 1:size(dp, 2)
            u = unique(dp[:, k])
            sort!(u)
            for i in eachindex(u)
                y = dp[:, k] .== u[i]
                x = xc[:, j]
                s = "$(vnames[k])=$(u[i])"
                pos = (.!ismissing.(y)) .& (.!ismissing.(x))
                if sum(pos) > 10
                    if min(std(x[pos]), std(y[pos])) > 1e-2
                        r = cor(y[pos], x[pos])
                        n = sum(skipmissing(y))
                        z = r * sqrt(length(collect(skipmissing(y))))
                        row = (Group=s, Component=j, N=n, Corr=r, Z=z)
                        push!(dr, row)
                    end
                end
            end
        end
    end

    write(out, string(dr))
    write(out, "\n\n")

end

function rungroup(k, df, ifig)

    df = select(df, Not("Response_Id"))
    vn = names(df)
    vn = [replace(x, "know_" => "k_") for x in vn]
    vn = [replace(x, "internet_" => "i_") for x in vn]
    rename!(df, vn)

    for c in names(df)
        if eltype(df[:, c]) <: Float64
            df[!, c] = Int.(df[:, c])
        end
    end

    # Passive variables
    dp = dmp[k]
    dp = select(dp, Not("Response_Id"))

    println("Active variables: ", names(df))
    println("Passive variables: ", names(dp))
    println("")

    mca = fit(MCA, df; method = "indicator", normalize = "principal", d = 2)
    px = quali_passive(mca, dp)
    write(out, @sprintf("==== %s ====\n\n", tm[k]))
    show(out, MIME("text/plain"), mca)
    cc = variable_coords(mca)
    cc = DataFrame(
        :Variable => cc.Variable,
        :Level => cc.Level,
        :X1 => cc.Coord[:, 1],
        :X2 => cc.Coord[:, 2],
    )

    xc = object_coords(mca)
    passive_summary(out, xc, dp)

    #write(out, "Difference in variable scores between first and last response level\n")
    #xx = combine(
    #    groupby(cc, :Variable),
    #    :X1 => x -> first(x) - last(x),
    #    :X2 => x -> first(x) - last(x),
    #)
    #write(out, string(xx))
    #write(out, "\n\n")

    PyPlot.clf()
    PyPlot.figure(figsize = (10, 8))
    PyPlot.grid(true)
    PyPlot.title("$(tm[k]) (n=$(size(df, 1)))")
    frst = Dict("i" => true, "k" => true, "a" => true)
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

    # Plot passive variables
    dpa = DataFrame(
        :Variable => px.Variable,
        :Level => px.Level,
        :Coord1 => px.Coord[:, 1],
        :Coord2 => px.Coord[:, 2],
    )
    for dz in groupby(dpa, :Variable)
        vname = first(dz[:, :Variable])
        for j = 1:size(dz, 1)
            level = dz[j, :Level]
            if count(skipmissing(dp[:, vname] .== level)) >= 10
                PyPlot.text(dz[j, :Coord1], dz[j, :Coord2], @sprintf("%s", level))
            end
        end
    end

    ha, lb = PyPlot.gca().get_legend_handles_labels()
    leg = PyPlot.figlegend(ha, lb, "center right")
    leg.draw_frame(false)
    PyPlot.xlabel("Component 1", size = 15)
    PyPlot.ylabel("Component 2", size = 15)
    PyPlot.savefig(@sprintf("plots/%03d.pdf", ifig))
    ifig += 1

    return ifig
end

function main(ifig)
    for (k, df) in dm
        ifig = rungroup(k, df, ifig)
    end
    return ifig
end

ifig = 0
ifig = main(ifig)

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
