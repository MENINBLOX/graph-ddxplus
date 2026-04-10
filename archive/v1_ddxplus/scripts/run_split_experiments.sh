#!/bin/bash
# 4개 그룹으로 분할해서 동시 실행

cd /home/max/Graph-DDXPlus

echo "Starting 4 parallel experiment groups..."

# 그룹 1: min_il (26개) - ports 7687,7688
nohup uv run python -u scripts/experiment_group.py --group 1 > results/group1_min_il.log 2>&1 &
echo "Group 1 (min_il): PID $!"

# 그룹 2: confidence, entropy, info_gain (31개) - ports 7689,7690
nohup uv run python -u scripts/experiment_group.py --group 2 > results/group2_conf_entropy.log 2>&1 &
echo "Group 2 (conf/entropy/ig): PID $!"

# 그룹 3: rank_stability, evidence_coverage (22개) - ports 7691,7692
nohup uv run python -u scripts/experiment_group.py --group 3 > results/group3_rank_evid.log 2>&1 &
echo "Group 3 (rank/evidence): PID $!"

# 그룹 4: disease_narrowing, confidence_stability, next_question_quality (48개) - ports 7693,7694
nohup uv run python -u scripts/experiment_group.py --group 4 > results/group4_disease_conf_next.log 2>&1 &
echo "Group 4 (disease/confstab/next): PID $!"

echo ""
echo "All groups started. Monitor with:"
echo "  tail -f results/group*.log"
