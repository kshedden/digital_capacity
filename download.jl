
DBU = "/home/kshedden/bin/dropbox_uploader.sh"

target = "/home/kshedden/data/Tawanna_Dillahunt/nsf.xlsx"

pa = "CTW|CEDER Quantitative Work/4-Data Collection/3-merge_data/output"
fn = "[NSF - Community Digital Capacity] - Final_August 8, 2022_12.28.xlsx"
ff = joinpath(pa, fn)

cmd = `$DBU download $(ff)`
run(cmd)

cmd = `mv $(fn) $target`
run(cmd)
