# The input is the file with metonymic pairs of the form:
# Hit Uppsala (disambiguation): (<Uppsala>, <Uppsala University>)

# This script will then split the input file column-wise
# into three separate files, namely, anchors, vehicles and targets.

log=$1

grep --color=never -P '^Hit.+(<.+>, <.+>)' log > newlog

awk -F ':'   '{print $1}' newlog > anchors
awk -F '>|<' '{print $2}' newlog  > vehicles
awk -F '>|<' '{print $4}' newlog > targets
sed 's/^Hit //' anchors > tmp
mv tmp anchors

rm newlog
