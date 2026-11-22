#!/bin/bash
# Tableau de bord des runs : bash ~/AA-Hackathon-EDTH-202611/fleet/status.sh
cd "$(dirname "$0")"
printf "%-10s %-13s %-5s %10s %5s %6s %6s %6s %8s\n" machine run tmux pas fps etape succes crash ep_rew
while read h n _; do
  ssh -o BatchMode=yes -o ConnectTimeout=5 -n $h "
    L=\$(ls -t /Data/arthur.fournier/fpv_runs/*.log 2>/dev/null | grep -v uv_sync | head -1)
    t=\$(tmux has-session -t train 2>/dev/null && echo oui || echo NON)
    g() { grep -E \"\\|    \$1 \" \$L 2>/dev/null | tail -1 | awk -F'|' '{gsub(/ /,\"\",\$3); print \$3}'; }
    printf '%-10s %-13s %-5s %10s %5s %6s %6s %6s %8s\n' $h $n \$t \"\$(g total_timesteps)\" \"\$(g fps)\" \"\$(g stage)\" \"\$(g success_rate)\" \"\$(g collision_rate)\" \"\$(g ep_rew_mean)\"
    grep -E 'Traceback|Error' \$L 2>/dev/null | tail -1 | sed 's/^/    !! /'
    grep '\[curriculum\]' \$L 2>/dev/null | tail -1 | sed 's/^/    /'
  " 2>/dev/null || echo "$h injoignable"
done < runs.txt
