#!/bin/bash
# 8개 Neo4j 인스턴스 설정

set -e

echo "1. 기존 컨테이너 중지..."
docker stop grcot-neo4j 2>/dev/null || true

echo "2. 볼륨 생성 및 데이터 복사..."
SOURCE_VOL="graph-ddxplus_neo4j_data"

for i in {1..8}; do
    VOL_NAME="graph-ddxplus_neo4j_data_$i"

    # 볼륨이 이미 있으면 삭제
    docker volume rm $VOL_NAME 2>/dev/null || true

    # 볼륨 생성
    docker volume create $VOL_NAME

    echo "  - 복사 중: $SOURCE_VOL -> $VOL_NAME"

    # 임시 컨테이너로 데이터 복사
    docker run --rm \
        -v $SOURCE_VOL:/source:ro \
        -v $VOL_NAME:/dest \
        alpine sh -c "cp -a /source/. /dest/"
done

echo "3. Docker Compose로 8개 컨테이너 시작..."
cd /home/max/Graph-DDXPlus
docker compose -f docker-compose.benchmark.yml up -d

echo "4. 컨테이너 헬스체크 대기..."
sleep 30

echo "5. 상태 확인..."
docker ps | grep neo4j-bench

echo ""
echo "완료! 포트 7687~7694에서 8개 Neo4j 실행 중"
