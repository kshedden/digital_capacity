using CSV
using DataFrames

pa = "/home/kshedden/data/Tawanna_Dillahunt"

fn = [
    "Michiganders.csv.gz",
    "Detroiters.csv.gz",
    "Non-Michiganders.csv.gz",
    "Parkside-Residents.csv.gz",
    "All-Valid-Responses.csv.gz",
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
    "allvalid" => copy(dx[5]),
)

demog = open(joinpath(pa, "demog.csv.gz")) do io
    CSV.read(io, DataFrame)
end

demog[:, :agegrp] = 10 * floor.(demog[:, :age] / 10)
demog[!, :agegrp] = Vector{Union{Int,Missing}}(demog[:, :agegrp])
demogv = [:sex, :agegrp, :age, :race, :education, :money, :hhs]
demog = demog[:, vcat(:Response_Id, demogv)]

# Align demographic data with capacity data.
dmp = Dict()
for k in keys(dm)
    dx = leftjoin(dm[k], demog, on = :Response_Id)
    dmp[k] = dx[:, vcat(:Response_Id, demogv)]
end
