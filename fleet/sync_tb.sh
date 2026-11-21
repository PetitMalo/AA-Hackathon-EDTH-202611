#!/bin/bash
# Rapatrie le tb/ du run le plus récent de chaque machine sur venturi : /Data/arthur.fournier/fpv_tb/<variante>/
cd "$(dirname "$0")"
DST=/Data/arthur.fournier/fpv_tb; mkdir -p $DST
while read h n _; do
  d=$(ssh -o BatchMode=yes -n $h 'ls -td /Data/arthur.fournier/fpv_runs/*/tb 2>/dev/null | head -1')
  [ -n "$d" ] && rsync -a -e "ssh -o BatchMode=yes" "$h:$d/" "$DST/$n/" &
done < runs.txt
wait; du -sh $DST
