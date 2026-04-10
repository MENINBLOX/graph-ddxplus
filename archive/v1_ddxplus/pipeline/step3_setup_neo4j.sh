#!/bin/bash
# Step 3: Neo4j 다중 인스턴스 설정 (병렬 실험용)
#
# 마스터 Neo4j(port 7687)의 데이터를 8개 인스턴스로 복제.
# 이를 통해 134,529건을 8개 워커로 병렬 처리 가능.

set -e
cd "$(dirname "$0")/.."

echo "============================================"
echo "Step 3: Neo4j Multi-Instance Setup"
echo "============================================"

# 마스터 Neo4j 확인
echo ""
echo "[1/3] 마스터 Neo4j 확인..."
if ! docker compose ps neo4j 2>/dev/null | grep -q "running"; then
    echo "  마스터 Neo4j 시작..."
    docker compose up -d neo4j
    echo "  대기 (10초)..."
    sleep 10
fi

# KG 검증
echo "  KG 검증..."
NODES=$(docker compose exec neo4j cypher-shell -u neo4j -p password123 \
    "MATCH (n) RETURN count(n) AS cnt" 2>/dev/null | tail -1 | tr -d ' ')
echo "  노드 수: ${NODES}"

if [ "$NODES" -lt "100" ]; then
    echo "  [오류] KG가 비어 있습니다. Step 1을 먼저 실행하세요."
    exit 1
fi

# 마스터 중지 (볼륨 복사를 위해)
echo ""
echo "[2/3] 볼륨 복제..."
docker compose stop neo4j

SOURCE_VOL="graph-ddxplus_neo4j_data"

for i in $(seq 1 8); do
    DEST_VOL="graph-ddxplus_neo4j_benchmark_${i}_data"
    echo "  복제: ${SOURCE_VOL} → ${DEST_VOL}"

    # 볼륨 생성 (이미 있으면 삭제 후 재생성)
    docker volume rm "$DEST_VOL" 2>/dev/null || true
    docker volume create "$DEST_VOL" > /dev/null

    # 데이터 복사
    docker run --rm \
        -v "${SOURCE_VOL}:/source:ro" \
        -v "${DEST_VOL}:/dest" \
        alpine sh -c "cp -a /source/. /dest/"
done

# 벤치마크 인스턴스 시작
echo ""
echo "[3/3] 벤치마크 인스턴스 시작..."
docker compose -f docker-compose.benchmark.yml up -d

echo ""
echo "  대기 (15초)..."
sleep 15

# 검증
echo ""
echo "=== 포트 확인 ==="
for port in 7687 7688 7689 7690 7691 7692 7693 7694; do
    STATUS=$(docker compose -f docker-compose.benchmark.yml exec \
        neo4j-$((port - 7686)) cypher-shell -u neo4j -p password123 \
        -a "bolt://localhost:${port}" \
        "RETURN 'OK'" 2>/dev/null | tail -1 | tr -d ' "' || echo "FAIL")
    echo "  port ${port}: ${STATUS}"
done

echo ""
echo "완료! 8개 Neo4j 인스턴스 (port 7687-7694)"
