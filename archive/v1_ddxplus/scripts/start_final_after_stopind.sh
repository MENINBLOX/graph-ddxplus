#!/bin/bash
# stopind 완료 대기 후 final 240 실행

echo "stopind 완료 대기 중..."
while ps aux | grep "run_stopping_independent_all" | grep -v grep > /dev/null 2>&1; do
    sleep 30
done

echo "stopind 완료! 10초 후 final 240 시작..."
sleep 10

chmod +x /home/max/Graph-DDXPlus/scripts/run_final_240.sh
bash /home/max/Graph-DDXPlus/scripts/run_final_240.sh
