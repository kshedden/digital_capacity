using CSV
using DataFrames

pa = "/home/kshedden/data/Tawanna_Dillahunt"

fn = [
    "Michiganders.csv.gz",
    "Detroiters.csv.gz",
    "Non-Michiganders.csv.gz",
    "Parkside-Residents.csv.gz",
]

dx = []
for f in fn
    df = open(joinpath(pa, f)) do io
        CSV.read(io, DataFrame)
    end
    push!(dx, df)
end

dm = Dict(
    "mich" => copy(dx[1]),
    "detroit" => copy(dx[2]),
    "nonmich" => copy(dx[3]),
    "parkside" => copy(dx[4]),
)

# Create region indicators
for j = 1:4
    dx[j][:, :reg] .= ["M", "D", "N", "P"][j]
end

da = vcat(dx...)
dm["all"] = da
