# IFTB Operations Runbook

## Overview

이 문서는 IFTB (Intelligent Futures Trading Bot) 운영을 위한 핵심 절차를 제공합니다.

## Table of Contents

1. [시스템 아키텍처](#시스템-아키텍처)
2. [긴급 절차](#긴급-절차)
3. [롤백 절차](#롤백-절차)
4. [재해 복구 (DR)](#재해-복구-dr)
5. [모니터링 및 알림](#모니터링-및-알림)
6. [일반 운영 절차](#일반-운영-절차)

---

## 시스템 아키텍처

```
                    ┌──────────────────┐
                    │   Binance API    │
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────┴─────────┐         ┌────────┴────────┐
    │  Data Collector   │         │  Order Executor │
    │  (WebSocket/REST) │         │  (CCXT)         │
    └─────────┬─────────┘         └────────┬────────┘
              │                             │
    ┌─────────┴─────────────────────────────┴─────────┐
    │                   IFTB Core                      │
    │  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
    │  │  Technical │  │    LLM     │  │  XGBoost   │ │
    │  │  Analysis  │  │  Analysis  │  │    ML      │ │
    │  │   (40%)    │  │   (25%)    │  │   (35%)    │ │
    │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘ │
    │        └───────────────┼───────────────┘        │
    │                  ┌─────┴─────┐                  │
    │                  │  Decision │                  │
    │                  │  Engine   │                  │
    │                  └─────┬─────┘                  │
    │                        │                        │
    │         ┌──────────────┼──────────────┐        │
    │         │              │              │        │
    │    ┌────┴────┐   ┌─────┴─────┐  ┌────┴────┐  │
    │    │ Circuit │   │   Risk    │  │  Kill   │  │
    │    │ Breaker │   │ Manager   │  │ Switch  │  │
    │    └─────────┘   └───────────┘  └─────────┘  │
    └──────────────────────────────────────────────────┘
              │                             │
    ┌─────────┴─────────┐         ┌────────┴────────┐
    │   PostgreSQL      │         │      Redis      │
    │   (Persistence)   │         │     (Cache)     │
    └───────────────────┘         └─────────────────┘
```

---

## 긴급 절차

### Kill Switch 활성화 (거래 즉시 중단)

**상황:** 비정상적인 시장 상황, 시스템 오류, 보안 문제 등으로 즉각적인 거래 중단이 필요할 때

```bash
# 방법 1: API 호출
curl -X POST http://localhost:8000/api/kill-switch \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"reason": "긴급 중단 사유"}'

# 방법 2: Docker 컨테이너 정지
docker stop iftb-bot

# 방법 3: 데이터베이스 플래그 설정
psql -h localhost -U iftb -d iftb -c "
  INSERT INTO kill_switch_events (activated_at, reason)
  VALUES (NOW(), '긴급 중단 사유');
"
```

**Kill Switch 비활성화:**
```bash
# 확인 코드 필요 (활성화 시 생성됨)
curl -X POST http://localhost:8000/api/kill-switch/deactivate \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"confirmation_code": "ABC123"}'

# 긴급 우회 (Admin Key 필요)
curl -X POST http://localhost:8000/api/kill-switch/force-deactivate \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"admin_key": "$KILL_SWITCH_ADMIN_KEY"}'
```

### Circuit Breaker 상태 확인

```bash
# 상태 확인
curl http://localhost:8000/api/circuit-breaker/status

# 수동 리셋 (HALF_OPEN 또는 OPEN 상태에서)
curl -X POST http://localhost:8000/api/circuit-breaker/reset \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

### 모든 포지션 긴급 청산

```bash
# 모든 오픈 포지션 청산
curl -X POST http://localhost:8000/api/positions/close-all \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"reason": "긴급 청산"}'
```

---

## 롤백 절차

### 1. 애플리케이션 롤백

```bash
# 현재 배포 버전 확인
kubectl get deployment iftb -o jsonpath='{.spec.template.spec.containers[0].image}'

# 이전 버전으로 롤백
kubectl rollout undo deployment/iftb

# 특정 버전으로 롤백
kubectl rollout undo deployment/iftb --to-revision=<revision-number>

# 롤백 상태 확인
kubectl rollout status deployment/iftb
```

### 2. Docker Compose 롤백

```bash
# 현재 이미지 태그 확인
docker inspect iftb-bot | jq '.[0].Config.Image'

# 이전 이미지로 롤백
export IFTB_IMAGE_TAG="previous-tag"
docker-compose down
docker-compose pull
docker-compose up -d

# 롤백 확인
docker-compose ps
docker-compose logs -f iftb
```

### 3. 데이터베이스 롤백

**⚠️ 주의: 데이터베이스 롤백은 데이터 손실을 초래할 수 있습니다.**

```bash
# 최신 백업 확인
ls -la /backup/postgres/

# 마이그레이션 롤백 (한 단계)
psql -h localhost -U iftb -d iftb -f migrations/rollback/004_rollback.sql

# 특정 시점으로 복원 (PITR)
# 1. 현재 DB 중지
docker stop iftb-postgres

# 2. 데이터 디렉토리 백업
mv /var/lib/postgresql/data /var/lib/postgresql/data.bak

# 3. 백업에서 복원
pg_restore -h localhost -U iftb -d iftb /backup/postgres/iftb_backup_YYYYMMDD.dump

# 4. DB 재시작
docker start iftb-postgres
```

### 4. ML 모델 롤백

```bash
# 현재 모델 버전 확인
ls -la /app/models/

# 이전 모델로 롤백
cp /backup/models/xgboost_v1.2.joblib /app/models/xgboost_model.joblib

# 모델 리로드 (API 호출)
curl -X POST http://localhost:8000/api/ml/reload-model \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

---

## 재해 복구 (DR)

### DR 시나리오

| 시나리오 | RTO | RPO | 복구 절차 |
|----------|-----|-----|-----------|
| 애플리케이션 장애 | 5분 | 0 | 컨테이너 재시작 |
| 데이터베이스 장애 | 15분 | 5분 | 레플리카 승격 |
| 전체 인프라 장애 | 1시간 | 15분 | DR 사이트 활성화 |
| 데이터 손상 | 2시간 | 1시간 | PITR 복원 |

### DR 체크리스트

#### 1. 일일 백업 확인
```bash
# 백업 상태 확인
./scripts/check_backups.sh

# 예상 출력:
# ✅ PostgreSQL daily backup: 2024-01-17 03:00 UTC
# ✅ Redis snapshot: 2024-01-17 03:15 UTC
# ✅ ML models backup: 2024-01-17 03:30 UTC
```

#### 2. DR 사이트 활성화

```bash
# 1. DNS 전환
./scripts/dr/switch_dns.sh --to=dr-site

# 2. DR 사이트 데이터베이스 확인
./scripts/dr/verify_db_sync.sh

# 3. DR 사이트 애플리케이션 시작
kubectl --context=dr-cluster apply -f k8s/dr/

# 4. 상태 확인
./scripts/dr/health_check.sh --site=dr
```

#### 3. Primary 사이트 복구

```bash
# 1. 문제 해결 확인
./scripts/dr/verify_primary_health.sh

# 2. 데이터 동기화
./scripts/dr/sync_from_dr.sh

# 3. DNS 전환 (Primary로)
./scripts/dr/switch_dns.sh --to=primary

# 4. DR 사이트 대기 모드 전환
./scripts/dr/standby_mode.sh
```

### 백업 정책

| 항목 | 빈도 | 보존 기간 | 위치 |
|------|------|-----------|------|
| PostgreSQL Full | 일 1회 | 30일 | S3 + Glacier |
| PostgreSQL WAL | 연속 | 7일 | S3 |
| Redis Snapshot | 시간 1회 | 24시간 | S3 |
| ML Models | 버전당 | 90일 | S3 |
| 설정 파일 | 변경 시 | 무기한 | Git |

---

## 모니터링 및 알림

### 핵심 메트릭 대시보드

- **Grafana**: http://localhost:3000
  - Trading Dashboard: 거래 성과, P&L, 포지션
  - System Health: CPU, 메모리, 네트워크
  - Analysis Dashboard: 신호 정확도, 모델 성능

### 알림 임계값

| 메트릭 | Warning | Critical | 대응 |
|--------|---------|----------|------|
| Drawdown | 10% | 15% | Circuit Breaker 확인 |
| API Error Rate | 5% | 10% | 연결 상태 점검 |
| Decision Latency | 500ms | 1s | 시스템 리소스 확인 |
| Daily Loss | 3% | 5% | 자동 거래 중단 |

### 알림 채널

```yaml
# Telegram 알림 설정
alerts:
  telegram:
    enabled: true
    chat_id: "${TELEGRAM_CHAT_ID}"
    token: "${TELEGRAM_BOT_TOKEN}"
    levels:
      - critical
      - warning

  # Slack 웹훅 (선택)
  slack:
    enabled: false
    webhook_url: "${SLACK_WEBHOOK_URL}"
```

---

## 일반 운영 절차

### 시스템 시작

```bash
# 전체 스택 시작
docker-compose up -d

# 상태 확인
docker-compose ps
docker-compose logs -f iftb

# 헬스체크
curl http://localhost:8000/health
```

### 시스템 종료 (정상)

```bash
# 1. 새 거래 중단
curl -X POST http://localhost:8000/api/trading/pause

# 2. 오픈 포지션 확인
curl http://localhost:8000/api/positions

# 3. 필요시 포지션 청산
curl -X POST http://localhost:8000/api/positions/close-all

# 4. 시스템 종료
docker-compose down
```

### 로그 확인

```bash
# 실시간 로그
docker-compose logs -f iftb

# 특정 기간 로그
docker-compose logs --since="2024-01-17T00:00:00" iftb

# 에러 로그만
docker-compose logs iftb 2>&1 | grep -i error
```

### 성능 점검

```bash
# 메모리 사용량
docker stats iftb-bot

# 데이터베이스 연결
psql -h localhost -U iftb -d iftb -c "SELECT * FROM pg_stat_activity WHERE datname = 'iftb';"

# Redis 상태
redis-cli -h localhost -p 6379 INFO
```

---

## 연락처

| 역할 | 담당자 | 연락처 |
|------|--------|--------|
| Primary Oncall | - | - |
| Backup Oncall | - | - |
| 인프라 팀 | - | - |
| 개발 팀 | - | - |

---

*Last Updated: 2024-01-17*
