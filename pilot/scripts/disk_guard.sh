#!/usr/bin/env bash
# Pauses the writers (SIGSTOP) while the Windows drive holding the WSL disk has under LOW GB free and
# resumes them (SIGCONT) above HIGH GB, so a growing ext4.vhdx cannot fill C: and crash WSL.
LOW=12; HIGH=20
paused=0
while true; do
  free=$(df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9)
  pids=$(ps -eo pid,args | grep -E "[i]e_v2.py|[d]ict_ie.py|[b]uild_graph_v2.py|[p]igz|[c]ollect_v3.py|VLLM::EngineCore|[E]ngineCore" | awk '{print $1}')
  if [ "$free" -lt $LOW ] && [ $paused = 0 ]; then
    [ -n "$pids" ] && kill -STOP $pids; paused=1; echo "$(date +%F' '%T) PAUSED free=${free}G pids=$(echo $pids)"
  elif [ "$free" -ge $HIGH ] && [ $paused = 1 ]; then
    [ -n "$pids" ] && kill -CONT $pids; paused=0; echo "$(date +%F' '%T) RESUMED free=${free}G"
  fi
  sleep 30
done
