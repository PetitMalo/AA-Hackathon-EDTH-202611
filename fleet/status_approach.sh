#!/bin/bash
# Tableau de bord des runs d'approche : bash ~/AA-Hackathon-EDTH-202611/fleet/status_approach.sh
cd "$(dirname "$0")"
printf "%-10s %-24s %-4s %10s %5s %5s %6s %6s %6s %7s\n" machine run tmux pas fps etape succes crash t_succ ep_rew
while read h n _; do
  ssh -o BatchMode=yes -o ConnectTimeout=5 -n $h "
    L=/Data/arthur.fournier/fpv_runs/approach/$n.log
    t=\$(tmux has-session -t approach 2>/dev/null && echo oui || echo NON)
    g() { grep -E \"\\|    \$1 \" \$L 2>/dev/null | tail -1 | awk -F'|' '{gsub(/ /,\"\",\$3); print \$3}'; }
    printf '%-10s %-24s %-4s %10s %5s %5s %6s %6s %6s %7s\n' $h $n \$t \"\$(g total_timesteps)\" \"\$(g fps)\" \"\$(g stage)\" \"\$(g success_rate)\" \"\$(g collision_rate)\" \"\$(g t_success_s)\" \"\$(g ep_rew_mean)\"
    grep -E 'Traceback|Error' \$L 2>/dev/null | tail -1 | sed 's/^/    !! /'
    grep '\[curriculum\]' \$L 2>/dev/null | tail -1 | sed 's/^/    /'
  " 2>/dev/null || echo "$h injoignable"
done < runs_approach.txt
